"""
Semantic segmentation models (semseg task), all trained from scratch under the common protocol.

PSPNet-ResNet50     Zhao, Shi, Qi, Wang & Jia, "Pyramid Scene Parsing Network" (CVPR 2017); official PyTorch code
                    hszhao/semseg model/pspnet.py: dilated ResNet-50 OS8, PPM bins 1/2/3/6 (2048 -> 512 each), conv3x3
                    512 + BN + ReLU + dropout 0.1 + 1x1; aux head on layer3 (conv3x3 256), weight 0.4.

[Unified] no ImageNet weights (the papers all start from ImageNet), common init, no mean / std, 608 input with the
YOLO-paper augmentation, Adam 1e-4 constant, cross-entropy with ignore 255 on the logits upsampled to the input,
PreciseBN for BN models (micro batch 2), K = C + 1 classes.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models._utils import IntermediateLayerGetter

from detectors.semseg.common import SemSegBase


def cbr(cin, cout, k=3, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _PreciseBN:
    precise_bn, precise_bn_images, precise_bn_batch_size = True, 1600, 16


class PPM(nn.Module):
    def __init__(self, cin, red, bins):
        super().__init__()
        self.stages = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(b), cbr(cin, red, k=1)) for b in bins)

    def forward(self, x):
        h, w = x.shape[-2:]
        return torch.cat([x] + [F.interpolate(s(x), (h, w), mode="bilinear", align_corners=True)
                                for s in self.stages], 1)


class PSPNet(_PreciseBN, SemSegBase):
    aux_weight = 0.4

    def __init__(self, num_classes):
        super().__init__(num_classes)
        bb = resnet50(weights=None, replace_stride_with_dilation=[False, True, True])           # OS8
        self.backbone = IntermediateLayerGetter(bb, {"layer3": "aux", "layer4": "out"})
        self.ppm = PPM(2048, 512, (1, 2, 3, 6))
        self.cls = nn.Sequential(cbr(4096, 512), nn.Dropout2d(0.1), nn.Conv2d(512, self.K, 1))
        self.aux = nn.Sequential(cbr(1024, 256), nn.Dropout2d(0.1), nn.Conv2d(256, self.K, 1))

    def logits(self, x):
        f = self.backbone(x)
        return self.cls(self.ppm(f["out"])), (self.aux(f["aux"]) if self.training else None)


# ---------------------------------------------------------------- SegFormer (NVlabs mix_transformer.py)


# ---------------------------------------------------------------- ImageNet-initialised variants (vseg01)
class ImageNetR50:
    """[Unified] vseg01 (2026-10-31), video segmentation stage on KITTI-STEP: the ResNet-50 backbone starts from the
    torchvision ImageNet-1k weights (ResNet50_Weights.IMAGENET1K_V1, the original torchvision ResNet-50) as in the
    papers, and the 0..1 input is normalised with the ImageNet mean / std inside the model (SemSegBase.forward).
    Everything else (heads, loss, recipe, PreciseBN) is the same as the FL36 models; the head keeps the common init.
    common/runner.py calls load_imagenet() right after the common initialisation (training only)."""

    imagenet_norm = True

    def backbone_module(self):
        raise NotImplementedError

    def load_imagenet(self):
        from torchvision.models import ResNet50_Weights
        w = ResNet50_Weights.IMAGENET1K_V1
        sd = {k: v for k, v in w.get_state_dict(progress=True).items() if not k.startswith("fc.")}
        self.backbone_module().load_state_dict(sd, strict=True)   # dilation changes strides only, not weights
        return f"torchvision {w} ({len(sd)} tensors, ResNet-50 backbone)"


class PSPNetIn(ImageNetR50, PSPNet):
    def backbone_module(self):
        return self.backbone


MODELS = {"pspnet": PSPNet, "pspnet_in": PSPNetIn}   # [Release] this repository's models only


def make_build(name):
    def build_model(task, num_classes):
        return MODELS[name](num_classes)
    return build_model
