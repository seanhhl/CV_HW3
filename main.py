import os

# ==========================================
# 環境變數與初始設定 (必須在載入 cv2 與 PyTorch 前設定)
# ==========================================
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "mask_rcnn_medical.pth")
    SUBMISSION_SAVE_PATH = os.path.join(OUTPUT_DIR, "submission.json")
    
    # ------------------
    # 2. 執行流程
    # ------------------
    if os.path.exists(TRAIN_DIR):
        print("Starting training...")
        trained_model = train_model(
            train_dir=TRAIN_DIR, 
            num_epochs=30, 
            batch_size=1, 
            save_path=MODEL_SAVE_PATH
        )
        
        if os.path.exists(TEST_DIR) and os.path.exists(JSON_MAPPING):
            print("Generating submission...")
            generate_submission(
                model=trained_model, 
                test_dir=TEST_DIR, 
                json_mapping_path=JSON_MAPPING, 
                output_path=SUBMISSION_SAVE_PATH
            )
    else:
        print(f"請確認 {TRAIN_DIR} 路徑是否正確上傳至 Colab。")