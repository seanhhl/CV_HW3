import os
import gc
import torch

# ==========================================
# 環境變數與初始設定 (必須在載入 cv2 與 PyTorch 前設定)
# ==========================================
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
# 增加 max_split_size_mb:128 避免記憶體碎片化造成的 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import cv2

try:
    cv2.setLogLevel(0)
    cv2.setNumThreads(0) 
except AttributeError:
    pass

# 匯入我們自己拆分的模組
from train import train_model
from inference import generate_submission

if __name__ == '__main__':
    # ------------------
    # 1. 路徑設定配置區
    # ------------------
    TRAIN_DIR = "./train"
    TEST_DIR = "./test_release"
    JSON_MAPPING = "./test_image_name_to_ids.json"
    
    # 設定輸出目錄 (Google Drive)
    OUTPUT_DIR = "/content/drive/MyDrive/Colab_files/CV_HW3"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 指定 Pretrained 權重與微調後儲存的路徑
    PRETRAINED_PATH = os.path.join(OUTPUT_DIR, "mask_r_cnn_pretrained.pth")
    FINETUNED_SAVE_PATH = os.path.join(OUTPUT_DIR, "mask_rcnn_medical_finetuned.pth")
    SUBMISSION_SAVE_PATH = os.path.join(OUTPUT_DIR, "submission_finetuned.json")
    
    # ------------------
    # 2. 執行流程
    # ------------------
    if os.path.exists(TRAIN_DIR):
        print("Starting Fine-tuning...")
        
        # [防 OOM 策略] 執行前強制垃圾回收與清空 CUDA 暫存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        trained_model = train_model(
            train_dir=TRAIN_DIR, 
            fine_tune_epochs=15,          # 指定微調 15 個 epoch
            batch_size=1, 
            pretrained_path=PRETRAINED_PATH, # 讀入你訓練好的 pretrained weight
            save_path=FINETUNED_SAVE_PATH    # 存成新的檔案以免覆蓋原有的 checkpoint
        )
        
        if os.path.exists(TEST_DIR) and os.path.exists(JSON_MAPPING):
            print("Generating submission from fine-tuned model...")
            
            # [防 OOM 策略] 訓練完成後，進入推論前再次清空記憶體
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            generate_submission(
                model=trained_model, 
                test_dir=TEST_DIR, 
                json_mapping_path=JSON_MAPPING, 
                output_path=SUBMISSION_SAVE_PATH
            )
    else:
        print(f"請確認 {TRAIN_DIR} 路徑是否正確上傳至 Colab。")