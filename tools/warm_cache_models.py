import os
import sys
import torch

# timm for ViT
try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

# torchvision for CNN backbones
import torchvision.models as tvm
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    RegNet_X_400MF_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
)


def info_header():
    print("== Warm Cache for Model_Compare ==")
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("Torchvision:", __import__('torchvision').__version__)
    print("timm:", __import__('timm').__version__ if HAS_TIMM else "not installed")
    print("HF_ENDPOINT:", os.environ.get('HF_ENDPOINT'))
    print("HF_HOME:", os.environ.get('HF_HOME'))
    # Torch hub cache dir
    try:
        from torch.hub import get_dir
        print("TORCH_HUB_DIR:", get_dir())
    except Exception:
        print("TORCH_HUB_DIR: unknown")


def warm_timm_vit():
    if not HAS_TIMM:
        print("[timm] timm not installed, skip ViT warmup")
        return
    print("[timm] warming vit_base_patch16_224 (pretrained=True) via HF")
    # Create model exactly as in ViT16_224.py
    _ = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1,
        in_chans=1,
        img_size=1024,
    )
    print("[timm] ViT cached")


def warm_torchvision_models():
    print("[torchvision] warming pre-trained CNN backbones")
    # Instantiate models with the exact weights enums used in code
    _ = tvm.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    print(" - densenet121 cached")

    _ = tvm.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    print(" - efficientnet_b0 cached")

    _ = tvm.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    print(" - mobilenet_v2 cached")

    _ = tvm.regnet_x_400mf(weights=RegNet_X_400MF_Weights.IMAGENET1K_V1)
    print(" - regnet_x_400mf cached")

    _ = tvm.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    print(" - resnet18 cached")

    _ = tvm.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    print(" - resnet34 cached")

    _ = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    print(" - resnet50 cached")

    _ = tvm.vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    print(" - vgg11 cached")

    _ = tvm.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
    print(" - vgg13 cached")

    _ = tvm.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    print(" - vgg16 cached")


def list_cache_summary():
    print("\n== Cache Summary ==")
    # Torchvision hub checkpoints
    hub_dir = None
    try:
        from torch.hub import get_dir
        hub_dir = get_dir()
        ckpt_dir = os.path.join(hub_dir, 'checkpoints')
        print("Torch hub checkpoints:", ckpt_dir)
        if os.path.isdir(ckpt_dir):
            for fn in sorted(os.listdir(ckpt_dir)):
                print("  ", fn)
        else:
            print("  (directory not found)")
    except Exception as e:
        print("Torch hub dir error:", repr(e))

    # HuggingFace hub
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    print("HF home:", hf_home)
    hf_hub_dir = os.path.join(hf_home, 'hub')
    print("HF hub dir:", hf_hub_dir)
    if os.path.isdir(hf_hub_dir):
        # list top-level model repos cached
        for d in sorted(os.listdir(hf_hub_dir)):
            print("  ", d)
    else:
        print("  (directory not found)")


def main():
    torch.set_grad_enabled(False)
    info_header()
    warm_torchvision_models()
    warm_timm_vit()
    list_cache_summary()


if __name__ == '__main__':
    main()
