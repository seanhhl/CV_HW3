import os
import cv2

# 設定環境變數以隱藏 OpenCV 底層的 C++ 警告
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

import json
import numpy as np
import torch
from pycocotools import mask as maskUtils
from dataset import MedicalCellDataset

def generate_submission(model, test_dir, json_mapping_path, output_path="submission.json"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    
    with open(json_mapping_path, 'r') as f:
        raw_mapping = json.load(f)

    # 處理 JSON 格式轉換
    name_to_id = {}
    if isinstance(raw_mapping, dict):
        name_to_id = raw_mapping
    elif isinstance(raw_mapping, list):
        for item in raw_mapping:
            if isinstance(item, dict):
                name_key = next((k for k in item.keys() if 'name' in k.lower() or 'file' in k.lower()), None)
                id_key = next((k for k in item.keys() if 'id' in k.lower()), None)
                if name_key and id_key:
                    name_to_id[str(item[name_key])] = int(item[id_key])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name_to_id[str(item[0])] = int(item[1])

    test_dataset = MedicalCellDataset(root_dir=test_dir, is_train=False, name_to_id_map=name_to_id)
    submission_results = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img_tensor, image_id = test_dataset[i]
            
            with torch.autocast(device_type='cuda'):
                prediction = model([img_tensor.to(device)])[0]
            
            masks = prediction['masks'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            boxes = prediction['boxes'].cpu().numpy()

            # 觀察 Score 分佈
            if len(scores) > 0:
                top_10_scores = scores[:10]
                avg_score = np.mean(scores)
                print(f"Image {image_id} - Top 10 Scores: {np.round(top_10_scores, 4)}")
                print(f"Image {image_id} - Avg Score: {avg_score:.4f}, Total detected: {len(scores)}")

            # ==========================================
            # [關鍵修改] 將門檻調降至 0.15 
            # 配合 V2 模型特性，挽救被誤殺的真實預測
            # ==========================================
            confidence_threshold = 0.15
            
            for m, label, score, box in zip(masks, labels, scores, boxes):
                if score >= confidence_threshold:
                    binary_mask = (m[0] > 0.5).astype(np.uint8)
                    xmin, ymin, xmax, ymax = box
                    bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                    
                    rle = maskUtils.encode(np.asfortranarray(binary_mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    
                    submission_results.append({
                        "image_id": int(image_id),
                        "bbox": bbox,
                        "score": float(score),
                        "category_id": int(label),
                        "segmentation": {
                            "size": rle["size"],
                            "counts": rle["counts"]
                        }
                    })

            del img_tensor, prediction
            torch.cuda.empty_cache()

    with open(output_path, 'w') as f:
        json.dump(submission_results, f, indent=2)
    print(f"Submission saved to {output_path}")