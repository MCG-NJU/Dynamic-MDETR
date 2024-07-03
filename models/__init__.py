from .dynamic_mdetr_resnet import DynamicMDETR as DynamicMDETR_ResNet
from .dynamic_mdetr_clip import DynamicMDETR as DynamicMDETR_CLIP

def build_model(args):
    assert args.model_type in ['ResNet', 'CLIP']
    if args.model_type == 'ResNet':
        return DynamicMDETR_ResNet(args)
    elif args.model_type == 'CLIP':
        return DynamicMDETR_CLIP(args)
