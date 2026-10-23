import time
import glob
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import concurrent.futures

BASE_DIR = Path("/home/turjo/Documents/Dataset_Final_Defence/model_showcase_website")
YOLO_ONNX = str(BASE_DIR / "models/checkpoints/best_yolo_int8.onnx")
HYBRID_ONNX = str(BASE_DIR / "models/checkpoints/best_hybrid_160_int8.onnx")
TEST_DIR = str(BASE_DIR / "data/vehicle_dataset_v0.1.2/valid")

print("Initializing Extreme CPU Optimizations...")

# 1. Aggressive Resolution Reduction
UNET_SIZE = 160  # Dropped from 640 -> 320 -> 160 (massively reduces math)
YOLO_SIZE = 320  # Keep at 320 as the ONNX model is hardcoded to this size

# 2. Optimized ONNX Session
sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 2 # Limit to prevent thread contention when running in parallel
sess_opt.inter_op_num_threads = 2
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

unet_session = ort.InferenceSession(HYBRID_ONNX, sess_opt, providers=['CPUExecutionProvider'])
unet_input_name = unet_session.get_inputs()[0].name

yolo = YOLO(YOLO_ONNX, task="segment")

img_path = glob.glob(f"{TEST_DIR}/*.jpg")[0]
bgr_img = cv2.imread(img_path)

def preprocess_unet(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (UNET_SIZE, UNET_SIZE)).astype(np.float32) / 255.0
    resized -= np.array([0.485, 0.456, 0.406])
    resized /= np.array([0.229, 0.224, 0.225])
    return np.expand_dims(np.transpose(resized, (2, 0, 1)), 0).astype(np.float32)

def run_unet(tensor):
    return unet_session.run(None, {unet_input_name: tensor})[0]

def run_yolo(img):
    return next(yolo.predict(img, stream=True, verbose=False, imgsz=YOLO_SIZE))

tensor_np = preprocess_unet(bgr_img)

print("Running warmup...")
for _ in range(3):
    run_unet(tensor_np)
    run_yolo(bgr_img)

print("Running 20 iterations with Multi-Threading + Aggressive Downscaling...")
latencies = []

# 3. True CPU Multi-threading
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    for _ in range(20):
        t0 = time.perf_counter()
        
        # Dispatch both models to run simultaneously on different CPU cores
        future_unet = executor.submit(run_unet, tensor_np)
        future_yolo = executor.submit(run_yolo, bgr_img)
        
        # Wait for both to finish
        _ = future_unet.result()
        _ = future_yolo.result()
        
        latencies.append((time.perf_counter() - t0) * 1000)

avg = sum(latencies)/len(latencies)
print(f"========================================")
print(f"EXTREME OPTIMIZED PERFORMANCE (CPU)")
print(f"Average Total Latency: {avg:.2f}ms")
print(f"Estimated FPS: {1000/max(1e-3, avg):.2f}")
print(f"========================================")
