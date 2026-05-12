import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes):
    """
    建立並回傳 Mask R-CNN 模型
    """
    # ==========================================
    # [策略 1] 局部解凍 (Partial Fine-tuning)
    # 將 trainable_backbone_layers 設為 3 (這也是 PyTorch 的最佳實踐預設值)
    # 凍結前兩層基礎邊緣提取特徵，解凍後三層讓其適應醫療細胞的形狀
    # ==========================================
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        trainable_backbone_layers=3
    )

    # 取得 box predictor 的輸入特徵維度
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 替換 head，num_classes 包含背景 (1 背景 + 4 細胞 = 5)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 取得 mask predictor 的輸入特徵維度
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 替換 mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model