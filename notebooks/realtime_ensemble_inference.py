import os
import time
import glob
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from ultralytics import YOLO
import matplotlib.colors as mcolors

# --- Model Architectures ---

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
    def __init__(self, num_classes, pretrained=True):
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


# --- Real-Time Optimized Inference Pipeline ---

class RealTimeMasterSegmenter:
    def __init__(self, hybrid_weights_path: str, yolo_weights_path: str, num_classes: int = 8, semantic_img_size: int = 320, compile_model: bool = True):
        self.num_classes = num_classes
        self.semantic_img_size = semantic_img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = torch.cuda.is_available()
        
        print(f"[Init] Setting up Real-Time Ensemble on {self.device} (FP16: {self.fp16})")

        # 1. Load and optimize Semantic Model
        self.hybrid = ProposedHybridASPPCA(self.num_classes, pretrained=False).to(self.device)
        if os.path.exists(hybrid_weights_path):
            self.hybrid.load_state_dict(torch.load(hybrid_weights_path, map_location=self.device))
        else:
            print(f"Warning: Semantic weights not found at {hybrid_weights_path}")
        
        self.hybrid.eval()
        if self.fp16:
            self.hybrid.half()

        if compile_model and hasattr(torch, 'compile'):
            try:
                print("[Init] Compiling Semantic Model with TorchScript...")
                self.hybrid = torch.compile(self.hybrid, mode='reduce-overhead')
            except Exception as e:
                print(f"[Init] torch.compile failed ({e}), falling back to standard execution.")

        # 2. Load YOLO Model
        self.yolo = YOLO(yolo_weights_path)
        
        # 3. Setup Preprocessing & GPU Streams
        self.transform = A.Compose([
            A.Resize(self.semantic_img_size, self.semantic_img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        if self.fp16:
            self.stream_semantic = torch.cuda.Stream()
            self.stream_yolo = torch.cuda.Stream()
            
        # Pre-allocate instance probability tensor to save reallocation time
        # We'll initialize it lazily in predict_fused.
        self.probs_instance = None 

        self._warmup()

    def _warmup(self):
        print("[Init] Running 3 Warmup passes...")
        dummy_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        for _ in range(3):
            _ = self.predict_fused(dummy_bgr, use_clahe=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[Init] Warmup complete.")

    @torch.no_grad()
    def predict_fused(self, image_bgr: np.ndarray, alpha: float = 0.55, conf_thresh: float = 0.35, use_clahe: bool = False):
        t_start = time.perf_counter()
        h_orig, w_orig = image_bgr.shape[:2]

        # 1. CPU Preprocessing
        if use_clahe:
            lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            rgb = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image=rgb)["image"].unsqueeze(0).to(self.device, non_blocking=True)
        if self.fp16:
            tensor = tensor.half()

        # Initialize instance probabilities on VRAM lazily (if size changed or first run)
        if self.probs_instance is None or self.probs_instance.shape[1:] != (h_orig, w_orig):
            self.probs_instance = torch.zeros(
                (self.num_classes, h_orig, w_orig),
                dtype=torch.float16 if self.fp16 else torch.float32,
                device=self.device
            )
        
        # Reset instance probabilities
        self.probs_instance.zero_()
        self.probs_instance[0, :, :] = 0.5  # Neutral background baseline

        counts = {}
        detections = []
        
        # Parallel Execution using CUDA streams (if available)
        with torch.amp.autocast('cuda', enabled=self.fp16):
            if self.fp16:
                # --- Semantic Stream ---
                with torch.cuda.stream(self.stream_semantic):
                    logits = self.hybrid(tensor)
                    probs_semantic = F.softmax(logits, dim=1)
                    probs_semantic_resized = F.interpolate(
                        probs_semantic, size=(h_orig, w_orig), mode='bilinear', align_corners=False
                    ).squeeze(0)

                # --- YOLO Stream ---
                with torch.cuda.stream(self.stream_yolo):
                    yolo_res = next(self.yolo.predict(
                        source=image_bgr,
                        conf=conf_thresh,
                        imgsz=self.semantic_img_size, # Or keeping original size, but 320 for speed
                        device=self.device,
                        verbose=False,
                        stream=True
                    ))
                    
                    names = yolo_res.names
                    if yolo_res.boxes is not None and yolo_res.masks is not None and len(yolo_res.boxes) > 0:
                        masks_gpu = yolo_res.masks.data.unsqueeze(1)
                        if self.fp16:
                            masks_gpu = masks_gpu.half()
                        
                        masks_resized = F.interpolate(
                            masks_gpu, size=(h_orig, w_orig), mode='nearest'
                        ).squeeze(1)

                        for i, box in enumerate(yolo_res.boxes):
                            cls_id = int(box.cls[0].item())
                            cls_name = names.get(cls_id, f"class_{cls_id}")
                            conf = float(box.conf[0].item())

                            counts[cls_name] = counts.get(cls_name, 0) + 1
                            detections.append({
                                "class": cls_name,
                                "confidence": round(conf, 3),
                                "bbox": [round(c, 1) for c in box.xyxy[0].tolist()]
                            })

                            target_cls = cls_id + 1
                            if target_cls < self.num_classes:
                                conf_tensor = torch.tensor(conf, dtype=masks_resized.dtype, device=self.device)
                                self.probs_instance[target_cls] = torch.maximum(
                                    self.probs_instance[target_cls],
                                    masks_resized[i] * conf_tensor
                                )
                
                # Synchronize streams before fusion
                torch.cuda.synchronize()
                
            else:
                # Fallback sequential execution
                logits = self.hybrid(tensor)
                probs_semantic = F.softmax(logits, dim=1)
                probs_semantic_resized = F.interpolate(
                    probs_semantic, size=(h_orig, w_orig), mode='bilinear', align_corners=False
                ).squeeze(0)

                yolo_res = next(self.yolo.predict(
                    source=image_bgr,
                    conf=conf_thresh,
                    device=self.device,
                    verbose=False,
                    stream=True
                ))
                
                names = yolo_res.names
                if yolo_res.boxes is not None and yolo_res.masks is not None and len(yolo_res.boxes) > 0:
                    masks_gpu = yolo_res.masks.data.unsqueeze(1).float()
                    masks_resized = F.interpolate(
                        masks_gpu, size=(h_orig, w_orig), mode='nearest'
                    ).squeeze(1)

                    for i, box in enumerate(yolo_res.boxes):
                        cls_id = int(box.cls[0].item())
                        cls_name = names.get(cls_id, f"class_{cls_id}")
                        conf = float(box.conf[0].item())
                        
                        counts[cls_name] = counts.get(cls_name, 0) + 1
                        detections.append({
                            "class": cls_name,
                            "confidence": round(conf, 3),
                            "bbox": [round(c, 1) for c in box.xyxy[0].tolist()]
                        })
                        
                        target_cls = cls_id + 1
                        if target_cls < self.num_classes:
                            conf_tensor = torch.tensor(conf, dtype=masks_resized.dtype, device=self.device)
                            self.probs_instance[target_cls] = torch.maximum(
                                self.probs_instance[target_cls],
                                masks_resized[i] * conf_tensor
                            )

            # 4. Fusion in VRAM
            fused_probs = (alpha * probs_semantic_resized) + ((1.0 - alpha) * self.probs_instance)
            final_mask_gpu = torch.argmax(fused_probs, dim=0).to(torch.uint8)
            final_mask = final_mask_gpu.cpu().numpy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return final_mask, counts, detections, latency_ms


# --- Main Benchmark Execution ---

if __name__ == "__main__":
    BASE_DIR = Path("/home/turjo/Documents/Dataset_Final_Defence/model_showcase_website")
    HYBRID_WEIGHTS = str(BASE_DIR / "models/checkpoints/best_hybrid.pt")
    YOLO_WEIGHTS = str(BASE_DIR / "models/checkpoints/best_yolo.pt")
    TEST_DIR = str(BASE_DIR / "data/vehicle_dataset_v0.1.2/valid")
    OUTPUT_FILE = str(BASE_DIR / "results/realtime_sample_output.png")
    
    CLASS_NAMES = ['background', 'auto_rickshaw', 'bus', 'car', 'cng', 'leguna', 'motorcycle', 'truck']
    
    # Generate colormap
    colors = np.zeros((len(CLASS_NAMES), 3), dtype=np.uint8)
    for i in range(1, len(CLASS_NAMES)):
        rgb = mcolors.hsv_to_rgb([(i - 1) / max(1, (len(CLASS_NAMES) - 1)), 0.85, 0.95])
        colors[i] = (np.array(rgb) * 255).astype(np.uint8)
    
    print("\n[Start] Initializing RealTimeMasterSegmenter...")
    segmenter = RealTimeMasterSegmenter(
        hybrid_weights_path=HYBRID_WEIGHTS,
        yolo_weights_path=YOLO_WEIGHTS,
        num_classes=len(CLASS_NAMES),
        semantic_img_size=320,  # Lower resolution for massive latency improvement
        compile_model=True
    )
    
    # Get a sample image
    test_images = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    if not test_images:
        print(f"[Error] No JPG images found in {TEST_DIR}")
        exit(1)
        
    sample_img_path = test_images[0]
    bgr_img = cv2.imread(sample_img_path)
    
    if bgr_img is None:
        print(f"[Error] Failed to read image {sample_img_path}")
        exit(1)
        
    print(f"\n[Benchmark] Running 20 iterations on {Path(sample_img_path).name} ({bgr_img.shape[1]}x{bgr_img.shape[0]})...")
    
    latencies = []
    final_mask = None
    v_counts = {}
    v_dets = []
    
    # 5 warmup runs specifically for this image resolution
    for _ in range(5):
        segmenter.predict_fused(bgr_img, use_clahe=False)
        
    for i in range(20):
        f_mask, counts, dets, lat_ms = segmenter.predict_fused(bgr_img, use_clahe=False)
        latencies.append(lat_ms)
        if i == 0:
            final_mask, v_counts, v_dets = f_mask, counts, dets
            
    avg_latency = np.mean(latencies)
    fps = 1000.0 / max(1e-3, avg_latency)
    
    print("\n" + "="*50)
    print("🚀 PERFORMANCE REPORT")
    print("="*50)
    print(f"Target Latency: < 80ms (12+ FPS)")
    print(f"Average Latency: {avg_latency:.2f}ms")
    print(f"Estimated FPS: {fps:.2f} FPS")
    print(f"Min Latency: {np.min(latencies):.2f}ms")
    print(f"Max Latency: {np.max(latencies):.2f}ms")
    print(f"Objects Detected: {v_counts}")
    print("="*50)
    
    # Save output visualization
    color_mask = colors[final_mask]
    overlay = cv2.addWeighted(bgr_img, 0.6, color_mask, 0.4, 0)
    
    for det in v_dets:
        box = det["bbox"]
        cls_name = det["class"]
        conf = det["confidence"]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 128), 2)
        cv2.putText(overlay, f"{cls_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
        
    # Draw latency panel
    cv2.putText(overlay, f"Latency: {avg_latency:.1f}ms ({fps:.1f} FPS)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imwrite(OUTPUT_FILE, overlay)
    print(f"\n[Result] Sample output visualization saved to: {OUTPUT_FILE}")
