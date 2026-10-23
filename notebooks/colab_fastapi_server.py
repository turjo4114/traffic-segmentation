import os
import time
import io
import asyncio
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.colors as mcolors

# Import architectures directly here to ensure the notebook runs standalone
import torch.nn as nn
import timm

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1)
    def forward(self, x):
        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        return x * torch.sigmoid(self.conv_h(x_h)) * torch.sigmoid(self.conv_w(x_w.permute(0, 1, 3, 2)))

class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256, rates=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True))
        ])
        for r in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(True)
            ))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.project = nn.Sequential(
            nn.Conv2d((len(rates) + 2) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(True), nn.Dropout(0.3)
        )
    def forward(self, x):
        size = x.shape[-2:]
        res = [conv(x) for conv in self.convs]
        res.append(F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False))
        return self.project(torch.cat(res, dim=1))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, use_ca=True):
        super().__init__()
        self.ca = CoordinateAttention(skip_ch) if (skip_ch > 0 and use_ca) else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            if self.ca is not None:
                skip = self.ca(skip)
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ProposedHybridASPPCA(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model('resnet50', pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        chs = self.encoder.feature_info.channels()
        self.aspp = ASPP(chs[4], out_channels=256)
        self.b0 = DecoderBlock(256, chs[3], 256, use_ca=True)
        self.b1 = DecoderBlock(256, chs[2], 128, use_ca=True)
        self.b2 = DecoderBlock(128, chs[1], 64, use_ca=True)
        self.b3 = DecoderBlock(64, chs[0], 32, use_ca=True)
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    def forward(self, x):
        inp_size = x.shape[-2:]
        f = self.encoder(x)
        d = self.b0(self.aspp(f[4]), f[3])
        d = self.b1(d, f[2])
        d = self.b2(d, f[1])
        d = self.b3(d, f[0])
        d = F.interpolate(d, size=inp_size, mode='bilinear', align_corners=False)
        return self.final(d)


# --- Global Variables ---
app = FastAPI(title="SegFusion GPU Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FP16 = torch.cuda.is_available()
NUM_CLASSES = 8
SEMANTIC_SIZE = 320
CLASS_NAMES = ['background', 'auto_rickshaw', 'bus', 'car', 'cng', 'leguna', 'motorcycle', 'truck']

# GPU Tensors/Streams
hybrid_model = None
yolo_model = None
transform = None
stream_semantic = None
stream_yolo = None
probs_instance = None
colors = None

@app.on_event("startup")
async def load_models():
    global hybrid_model, yolo_model, transform, stream_semantic, stream_yolo, probs_instance, colors
    print(f"Loading Models to {DEVICE} (FP16: {FP16})...")
    
    colors = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    for i in range(1, NUM_CLASSES):
        rgb = mcolors.hsv_to_rgb([(i - 1) / max(1, (NUM_CLASSES - 1)), 0.85, 0.95])
        colors[i] = (np.array(rgb) * 255).astype(np.uint8)

    hybrid_model = ProposedHybridASPPCA(NUM_CLASSES).to(DEVICE)
    hybrid_model.load_state_dict(torch.load("best_hybrid.pt", map_location=DEVICE))
    hybrid_model.eval()
    if FP16:
        hybrid_model.half()
        print("Compiling U-Net...")
        hybrid_model = torch.compile(hybrid_model, mode='reduce-overhead')
        stream_semantic = torch.cuda.Stream()
        stream_yolo = torch.cuda.Stream()

    yolo_model = YOLO("best_yolo.pt")

    transform = A.Compose([
        A.Resize(SEMANTIC_SIZE, SEMANTIC_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    print("Backend Ready!")

def infer_frame(image_bgr: np.ndarray):
    global probs_instance
    t_start = time.perf_counter()
    h_orig, w_orig = image_bgr.shape[:2]

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(image=rgb)["image"].unsqueeze(0).to(DEVICE, non_blocking=True)
    if FP16: tensor = tensor.half()

    if probs_instance is None or probs_instance.shape[1:] != (h_orig, w_orig):
        probs_instance = torch.zeros((NUM_CLASSES, h_orig, w_orig), dtype=torch.float16 if FP16 else torch.float32, device=DEVICE)
    probs_instance.zero_()
    probs_instance[0, :, :] = 0.5  

    with torch.amp.autocast('cuda', enabled=FP16):
        if FP16:
            with torch.cuda.stream(stream_semantic):
                logits = hybrid_model(tensor)
                probs_semantic_resized = F.interpolate(F.softmax(logits, dim=1), size=(h_orig, w_orig), mode='bilinear', align_corners=False).squeeze(0)
            with torch.cuda.stream(stream_yolo):
                yolo_res = next(yolo_model.predict(image_bgr, conf=0.35, imgsz=SEMANTIC_SIZE, device=DEVICE, verbose=False, stream=True))
                if yolo_res.boxes is not None and yolo_res.masks is not None and len(yolo_res.boxes) > 0:
                    masks_resized = F.interpolate(yolo_res.masks.data.unsqueeze(1).half(), size=(h_orig, w_orig), mode='nearest').squeeze(1)
                    for i, box in enumerate(yolo_res.boxes):
                        cls_id = int(box.cls[0].item())
                        target_cls = cls_id + 1
                        if target_cls < NUM_CLASSES:
                            conf_tensor = torch.tensor(float(box.conf[0].item()), dtype=masks_resized.dtype, device=DEVICE)
                            probs_instance[target_cls] = torch.maximum(probs_instance[target_cls], masks_resized[i] * conf_tensor)
            torch.cuda.synchronize()
        else:
            # CPU Fallback
            probs_semantic_resized = F.interpolate(F.softmax(hybrid_model(tensor), dim=1), size=(h_orig, w_orig), mode='bilinear', align_corners=False).squeeze(0)
            yolo_res = next(yolo_model.predict(image_bgr, conf=0.35, device=DEVICE, verbose=False, stream=True))
            if yolo_res.boxes is not None and yolo_res.masks is not None and len(yolo_res.boxes) > 0:
                masks_resized = F.interpolate(yolo_res.masks.data.unsqueeze(1).float(), size=(h_orig, w_orig), mode='nearest').squeeze(1)
                for i, box in enumerate(yolo_res.boxes):
                    target_cls = int(box.cls[0].item()) + 1
                    if target_cls < NUM_CLASSES:
                        conf_tensor = torch.tensor(float(box.conf[0].item()), dtype=masks_resized.dtype, device=DEVICE)
                        probs_instance[target_cls] = torch.maximum(probs_instance[target_cls], masks_resized[i] * conf_tensor)

        fused_probs = (0.55 * probs_semantic_resized) + (0.45 * probs_instance)
        final_mask = torch.argmax(fused_probs, dim=0).to(torch.uint8).cpu().numpy()

    if FP16: torch.cuda.synchronize()
    latency = (time.perf_counter() - t_start) * 1000.0

    color_mask = colors[final_mask]
    overlay = cv2.addWeighted(image_bgr, 0.6, color_mask, 0.4, 0)
    
    if yolo_res.boxes is not None:
        for box in yolo_res.boxes:
            cls_name = yolo_res.names[int(box.cls[0].item())]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.putText(overlay, f"{cls_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)

    fps = 1000.0 / max(1e-3, latency)
    cv2.putText(overlay, f"Colab GPU: {latency:.1f}ms ({fps:.1f} FPS)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    return overlay

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"message": "Invalid Image"})
        
    overlay = infer_frame(img)
    _, encoded_img = cv2.imencode('.jpg', overlay)
    
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    print(f"==================================================")
    print(f"🚀 FASTAPI SERVER IS RUNNING!")
    print(f"To expose this to the internet with a STATIC URL, open a NEW Colab cell and run:")
    print(f"!npm install -g localtunnel")
    print(f"!lt --port 8000 --subdomain turjo-segfusion-backend")
    print(f"==================================================")
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
