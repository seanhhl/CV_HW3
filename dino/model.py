import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.detection import MaskRCNN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class DINOv2Backbone(nn.Module):
    """
    自定義 Backbone Wrapper，將 DINOv2 封裝成 Mask R-CNN 支援的格式
    已修改：完全凍結 DINOv2，不提供解凍功能
    """
    def __init__(self, model_name='dinov2_vits14', out_channels=256):
        super().__init__()
        
        # 1. 載入 DINOv2 模型
        print(f"Loading {model_name} from torch.hub...")
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', model_name)
        
        # 狀態：全數凍結 DINOv2 骨幹網路 (完全凍結)
        for param in self.dinov2.parameters():
            param.requires_grad = False
            
        embed_dim = self.dinov2.embed_dim
        self.out_channels = out_channels
        
        # 2. 定義多尺度 (FPN) 的目標 Stride
        self.scales = [4, 8, 16, 32, 64]
        
        # 3. 建立 FPN
        in_channels_list = [embed_dim] * len(self.scales)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 挑戰 1: 輸入尺寸防呆 (Padding)
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_pad = x
            
        # 由於已經不需要訓練骨幹網路，可以直接使用 forward_features 
        # 不需要計算梯度，如果是 inference 階段或 training 階段 backbone 都凍結
        ret = self.dinov2.forward_features(x_pad)
        patch_tokens = ret['x_norm_patchtokens'] 
        
        # 挑戰 2: 1D 序列轉 2D 特徵圖
        H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]
        h_feat, w_feat = H_pad // 14, W_pad // 14
        feat_2d = patch_tokens.transpose(1, 2).reshape(B, -1, h_feat, w_feat)
        
        # 挑戰 3: 單一尺度轉多尺度
        raw_features = OrderedDict()
        for i, stride in enumerate(self.scales):
            target_h = max(1, H // stride)
            target_w = max(1, W // stride)
            scaled_feat = F.interpolate(feat_2d, size=(target_h, target_w), mode='bilinear', align_corners=False)
            raw_features[str(i)] = scaled_feat
            
        out = self.fpn(raw_features)
        return out

def get_model_instance_segmentation(num_classes):
    """
    建立包含凍結 DINOv2 Backbone 的 Mask R-CNN 模型
    """
    backbone = DINOv2Backbone(model_name='dinov2_vits14', out_channels=256)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model
