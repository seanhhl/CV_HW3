import os
import torch
from torch.utils.data import DataLoader
from dataset import MedicalCellDataset, collate_fn
from model import get_model_instance_segmentation

def train_model(train_dir, num_epochs=30, batch_size=1, save_path='mask_rcnn_medical.pth', resume=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    dataset = MedicalCellDataset(root_dir=train_dir, is_train=True)
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:-20])
    
    data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, collate_fn=collate_fn, pin_memory=False
    )
    
    num_classes = 5 
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    
    # 保持 AdamW 與平滑學習率，這對局部解凍非常友善
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 0

    if resume and os.path.exists(save_path):
        print(f"發現存在的 Checkpoint：{save_path}，正在嘗試恢復訓練...")
        checkpoint = torch.load(save_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"成功載入！將從 Epoch {start_epoch + 1} 繼續訓練。")
        else:
            model.load_state_dict(checkpoint)
            print("載入舊版權重格式。")

    for epoch in range(start_epoch, num_epochs):
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
                
                # ==========================================
                # [策略 2] 強制加重 Mask Loss 權重
                # 給予 loss_mask 2.5 倍的權重，強迫模型專注於描繪細胞邊緣
                # ==========================================
                loss_weights = {
                    'loss_classifier': 1.0,
                    'loss_box_reg': 1.0,
                    'loss_mask': 2.5,  # 關鍵加權
                    'loss_objectness': 1.0,
                    'loss_rpn_box_reg': 1.0
                }
                
                # 計算加權後的總 Loss 供反向傳播使用
                losses = sum(loss_dict[k] * loss_weights.get(k, 1.0) for k in loss_dict.keys())

            scaler.scale(losses).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += losses.item()
            
            # 這裡印出的依然是「原始的 Loss 數值」方便我們觀察真實進展
            epoch_loss_cls += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
            epoch_loss_box += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
            epoch_loss_mask += loss_dict.get('loss_mask', torch.tensor(0.0)).item()
            epoch_loss_obj += loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
            epoch_loss_rpn += loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()
            
            del images, targets, loss_dict, losses

        lr_scheduler.step()
        
        num_batches = len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Weighted Total Loss: {epoch_loss/num_batches:.4f}")
        print(f"  -> Cls: {epoch_loss_cls/num_batches:.4f} | Box: {epoch_loss_box/num_batches:.4f} | Raw Mask: {epoch_loss_mask/num_batches:.4f} | Obj: {epoch_loss_obj/num_batches:.4f} | RPN: {epoch_loss_rpn/num_batches:.4f}")
        
        torch.cuda.empty_cache()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(checkpoint, save_path)
    
    return model