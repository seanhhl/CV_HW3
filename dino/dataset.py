import os

# 必須在 import cv2 之前設定環境變數
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import numpy as np
import torch
import cv2

# 進一步雙重確保靜音
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import albumentations as A

class MedicalCellDataset(Dataset):
    def __init__(self, root_dir, is_train=True, name_to_id_map=None):
        """
        root_dir: train/ 或 test_release/ 的路徑
        is_train: 布林值，判斷是否為訓練集
        name_to_id_map: 字典，提供檔名到 image_id 的映射 (主要用於生成 Submission)
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.name_to_id_map = name_to_id_map
        self.class_map = {'class1': 1, 'class2': 2, 'class3': 3, 'class4': 4}
        
        # 定義資料增強管線 (Data Augmentation Pipeline)
        self.transforms = None
        if self.is_train:
            self.image_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
            # 針對醫療影像細胞的特徵，配置 Albumentations 資料增強
            self.transforms = A.Compose([
                A.HorizontalFlip(p=0.5),              # 50% 機率水平翻轉
                A.VerticalFlip(p=0.5),                # 50% 機率垂直翻轉
                A.RandomRotate90(p=0.5),              # 50% 機率隨機旋轉 90 度
                A.RandomBrightnessContrast(p=0.2),    # 20% 機率改變亮度與對比 (模擬染色/光源差異)
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.tif')]

    def __len__(self):
        return len(self.image_dirs) if self.is_train else len(self.image_files)

    def __getitem__(self, idx):
        if self.is_train:
            img_dir_name = self.image_dirs[idx]
            img_dir = os.path.join(self.root_dir, img_dir_name)
            img_path = os.path.join(img_dir, "image.tif")
            
            # 使用 OpenCV 讀取影像並轉為 RGB
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"無法讀取影像: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            masks = []
            labels = []
            
            for cls_name, cls_id in self.class_map.items():
                mask_path = os.path.join(img_dir, f"{cls_name}.tif")
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if mask_img is None:
                        continue
                    
                    instance_ids = np.unique(mask_img)
                    instance_ids = instance_ids[instance_ids != 0]
                    
                    for inst_id in instance_ids:
                        obj_mask = (mask_img == inst_id).astype(np.uint8)
                        masks.append(obj_mask)
                        labels.append(cls_id)
            
            boxes = []
            valid_masks = []
            valid_labels = []

            # 計算 Bounding Box
            for i in range(len(masks)):
                pos = np.where(masks[i])
                if len(pos[0]) == 0:
                    continue # 略過空遮罩
                    
                # pos[1] 是 column (x), pos[0] 是 row (y)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                
                # 防呆：確保 Bounding Box 寬高大於 0 (Albumentations 的嚴格要求)
                if xmax <= xmin: xmax = xmin + 1
                if ymax <= ymin: ymax = ymin + 1
                
                boxes.append([xmin, ymin, xmax, ymax])
                valid_masks.append(masks[i])
                valid_labels.append(labels[i])

            # 如果這張圖片有細胞，則進行資料增強
            if len(boxes) > 0 and self.transforms is not None:
                augmented = self.transforms(
                    image=img, 
                    masks=valid_masks, 
                    bboxes=boxes, 
                    labels=valid_labels
                )
                img = augmented['image']
                valid_masks = augmented['masks']
                boxes = augmented['bboxes']
                valid_labels = augmented['labels']

            # 將影像轉換為 PyTorch Tensor
            img_tensor = F.to_tensor(img)

            # 封裝目標 (Targets)
            if len(boxes) == 0:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)
                masks_tensor = torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.uint8)
                area = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
            else:
                masks_tensor = torch.as_tensor(np.stack(valid_masks, axis=0), dtype=torch.uint8)
                boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.as_tensor(valid_labels, dtype=torch.int64)
                area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
                iscrowd = torch.zeros((len(boxes_tensor),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes_tensor
            target["labels"] = labels_tensor
            target["masks"] = masks_tensor
            target["image_id"] = torch.tensor([idx])
            target["area"] = area
            target["iscrowd"] = iscrowd

            return img_tensor, target

        else:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.root_dir, img_name)
            
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"無法讀取測試集影像: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(img)
            
            image_id = -1
            if self.name_to_id_map is not None:
                image_id = self.name_to_id_map.get(img_name, self.name_to_id_map.get(img_name.replace('.tif', ''), -1))
                
            return img_tensor, image_id

def collate_fn(batch):
    """處理 DataLoader 中不同大小的 target dict"""
    return tuple(zip(*batch))