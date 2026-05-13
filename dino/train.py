import os
import torch
from torch.utils.data import DataLoader
from dataset import MedicalCellDataset, collate_fn
from model import get_model_instance_segmentation

def train_model(train_dir, fine_tune_epochs=15, batch_size=1, pretrained_path='mask_r_cnn_pretrained.pth', save_path='mask_rcnn_medical_finetuned.pth'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    dataset = MedicalCellDataset(root_dir=train_dir, is_train=True)
    indices = torch.randperm(len(dataset)).tolist()
    # 提醒：目前這裡切出了 20 張圖，但沒有放入 Validation Dataloader 中使用
    train_dataset = torch.utils.data.Subset(dataset, indices[:-20])
    
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, collate_fn=collate_fn, pin_memory=False
    )
    
    num_classes = 5 
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    
    # ==========================================
    # [微調策略] 使用較小的學習率重新初始化 Optimizer
    # 為了避免破壞已經學好的特徵，微調時我們將 LR 從 2e-4 降至 5e-5
    # ==========================================
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    # 重新設定 15 Epochs 的 Cosine 排程
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0

    # 讀取 Pretrained Weights
    if os.path.exists(pretrained_path):
        print(f"發現 Pretrained Checkpoint：{pretrained_path}，正在載入權重...")
        checkpoint = torch.load(pretrained_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 關鍵：我們只載入 model_state_dict，不載入 optimizer，以套用新的學習率排程
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            print(f"成功載入權重！將從 Epoch {start_epoch} 開始接續微調 {fine_tune_epochs} 個 Epoch。")
        else:
            model.load_state_dict(checkpoint)
            print("載入舊版權重格式。")
    else:
        print(f"找不到 {pretrained_path}，將從 Epoch 0 從頭開始訓練。")

    # 計算目標 Epoch (例如原本跑到 30，微調 15，就會跑到 45)
    target_epoch = start_epoch + fine_tune_epochs

    for epoch in range(start_epoch, target_epoch):
        model.train()
        epoch_loss = 0
        epoch_loss_cls = 0
        epoch_loss_box = 0
        epoch_loss_mask = 0
        epoch_loss_obj = 0
        epoch_loss_rpn = 0
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                
                # 強制加重 Mask Loss 權重
                loss_weights = {
                    'loss_classifier': 1.0,
                    'loss_box_reg': 1.0,
                    'loss_mask': 1.0,
                    'loss_objectness': 1.0,
                    'loss_rpn_box_reg': 1.0
                }
                
                losses = sum(loss_dict[k] * loss_weights.get(k, 1.0) for k in loss_dict.keys())

            scaler.scale(losses).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
            epoch_loss_cls += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
            epoch_loss_box += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            epoch_loss_mask += loss_dict.get('loss_mask', torch.tensor(0.0)).item()
            epoch_loss_obj += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
            epoch_loss_rpn += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
            
            del images, targets, loss_dict, losses

        lr_scheduler.step()
        
        num_batches = len(data_loader)
        print(f"Epoch {epoch+1}/{target_epoch}, Weighted Total Loss: {epoch_loss/num_batches:.4f}")
        print(f"  -> Cls: {epoch_loss_cls/num_batches:.4f} | Box: {epoch_loss_box/num_batches:.4f} | Raw Mask: {epoch_loss_mask/num_batches:.4f} | Obj: {epoch_loss_obj/num_batches:.4f} | RPN: {epoch_loss_rpn/num_batches:.4f}")
        
        torch.cuda.empty_cache()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 儲存微調階段的 optimizer 狀態以防中斷
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, save_path)
    
    return model