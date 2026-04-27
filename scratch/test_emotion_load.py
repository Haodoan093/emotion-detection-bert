
import torch
from transformers import pipeline
import os

model_path = "models/emotion/bench/distil_uit_e3_focal/best"
print(f"Checking path: {os.path.abspath(model_path)}")
if os.path.exists(model_path):
    print("Path exists.")
else:
    print("Path does not exist!")

try:
    classifier = pipeline(
        "text-classification",
        model=model_path,
        device=-1, # CPU for test
        truncation=True,
    )
    result = classifier("Tôi rất lo lắng về tình hình tài chính của công ty.")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
