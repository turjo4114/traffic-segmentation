import time
import glob
from pathlib import Path
import cv2
from ultralytics import YOLO

BASE_DIR = Path("/home/turjo/Documents/Dataset_Final_Defence/model_showcase_website")
YOLO_WEIGHTS = str(BASE_DIR / "models/checkpoints/best_yolo.pt")
TEST_DIR = str(BASE_DIR / "data/vehicle_dataset_v0.1.2/valid")

print(f"Loading YOLO from {YOLO_WEIGHTS}...")
yolo = YOLO(YOLO_WEIGHTS)

img_path = glob.glob(f"{TEST_DIR}/*.jpg")[0]
img = cv2.imread(img_path)

print(f"Running 5 warmup passes...")
for _ in range(5):
    next(yolo.predict(img, stream=True, verbose=False))

print("Running 20 iterations...")
latencies = []
for _ in range(20):
    t0 = time.perf_counter()
    next(yolo.predict(img, stream=True, verbose=False))
    latencies.append((time.perf_counter() - t0) * 1000)

avg = sum(latencies)/len(latencies)
print(f"========================================")
print(f"YOLO-ONLY PERFORMANCE (CPU)")
print(f"Average Latency: {avg:.2f}ms")
print(f"Estimated FPS: {1000/max(1e-3, avg):.2f}")
print(f"========================================")
