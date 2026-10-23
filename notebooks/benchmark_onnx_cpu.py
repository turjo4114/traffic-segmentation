import time
import glob
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

BASE_DIR = Path("/home/turjo/Documents/Dataset_Final_Defence/model_showcase_website")
YOLO_ONNX = str(BASE_DIR / "models/checkpoints/best_yolo_int8.onnx")
HYBRID_ONNX = str(BASE_DIR / "models/checkpoints/best_hybrid_int8.onnx")
TEST_DIR = str(BASE_DIR / "data/vehicle_dataset_v0.1.2/valid")

print("Loading ONNX models...")
# YOLO
yolo = YOLO(YOLO_ONNX, task="segment")
# U-Net
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 4
unet_session = ort.InferenceSession(HYBRID_ONNX, sess_opt, providers=['CPUExecutionProvider'])
unet_input_name = unet_session.get_inputs()[0].name

img_path = glob.glob(f"{TEST_DIR}/*.jpg")[0]
bgr_img = cv2.imread(img_path)

# Resize for UNet
rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (320, 320))
# normalize like standard torchvision
resized = resized.astype(np.float32) / 255.0
resized -= np.array([0.485, 0.456, 0.406])
resized /= np.array([0.229, 0.224, 0.225])
tensor_np = np.transpose(resized, (2, 0, 1))
tensor_np = np.expand_dims(tensor_np, 0).astype(np.float32)

print("Running 3 warmup passes...")
for _ in range(3):
    next(yolo.predict(bgr_img, stream=True, verbose=False, imgsz=320))
    unet_session.run(None, {unet_input_name: tensor_np})

print("Running 20 iterations of full ensemble...")
latencies = []
for _ in range(20):
    t0 = time.perf_counter()
    # 1. Run YOLO (simulate concurrent execution by just measuring sequential but YOLO ONNX is fast)
    next(yolo.predict(bgr_img, stream=True, verbose=False, imgsz=320))
    # 2. Run U-Net
    unet_session.run(None, {unet_input_name: tensor_np})
    
    latencies.append((time.perf_counter() - t0) * 1000)

avg = sum(latencies)/len(latencies)
print(f"========================================")
print(f"ONNX INT8 ENSEMBLE PERFORMANCE (CPU)")
print(f"Average Total Latency: {avg:.2f}ms")
print(f"Estimated FPS: {1000/max(1e-3, avg):.2f}")
print(f"========================================")
