"""
Semantic segmentation task (FL36-Seg / VOC2007 segmentation subset), K = C + 1 classes (0 = background).

Model contract
    training  : loss, _ = model(images, {"sem": LongTensor [B, S, S]})   (255 = ignore)
    inference : model(images) -> logits [B, K, h, w], h / w proportional to the input (upsampled here)
Evaluation (original image pixels): logits of the letterboxed area -> bilinear resize to the original size ->
argmax -> confusion matrix (void 255 skipped). Reported: mIoU over all K classes (incl. background, the same
definition as the "derived semantic mIoU" of the instance models in tasks/seg_task.py), mIoU over the C object
classes, pixel accuracy. "miou" / "miou_fg" carry {"summary": {"mAP": value}} so the common runner records them
in the eval curve like detection mAP.

[Unified] vseg01 (2026-10-31): UNIFIED_SEG_DATASET=kittistep reads KITTI-STEP (common/seg/kitti_step.py) on the wide
canvas spec["img_h"] x img_size (1216 x 384). Background (index 0) has no ground truth there, so "mIoU (objects)" is
the 19-class KITTI-STEP mIoU. With img_h None (FL36 / VOC) everything is exactly as before.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from common import config as C
from common.datasets.voc import letterbox_params
from common.datasets.voc_semseg import VOCSemSeg
from common.mask_ops import miou_from_confusion, semantic_confusion
from common.progress import Progress
from common.trainer import make_loader


class SemSegTask:
    name = "semseg"

    def __init__(self):
        self.spec = C.TASKS["semseg"]
        self.class_names = list(self.spec["classes"])
        self.img_h = self.spec.get("img_h")

    def _kitti(self, args, split, train):
        from common.seg.kitti_step import KittiStepSemSeg
        return KittiStepSemSeg(C.SEG_ROOT, split, args.img_size, self.class_names, train=train,
                               hflip_prob=C.PROFILE["hflip_prob"] if train else 0.0,
                               max_samples=args.max_train if train else args.max_test,
                               aug=C.PROFILE["image_aug"] if train else None, img_h=self.img_h)

    def train_set(self, args):
        if self.spec["dataset"] == "kittistep":
            return self._kitti(args, self.spec["train_split"], True)
        return VOCSemSeg(C.SEG_ROOT, self.spec["train_split"], args.img_size, self.class_names, train=True,
                         hflip_prob=C.PROFILE["hflip_prob"], max_samples=args.max_train, aug=C.PROFILE["image_aug"])

    def test_set(self, args):
        if self.spec["dataset"] == "kittistep":
            return self._kitti(args, self.spec["test_split"], False)
        return VOCSemSeg(C.SEG_ROOT, self.spec["test_split"], args.img_size, self.class_names, train=False,
                         max_samples=args.max_test)

    @staticmethod
    def to_original(logits, img_size, w, h, img_h=None):
        """logits [K, h', w'] over the letterboxed input -> label map [h, w] (numpy int64)."""
        ih = img_h or img_size
        lg = F.interpolate(logits[None].float(), (ih, img_size), mode="bilinear", align_corners=False)[0]
        _, px, py, rw, rh = letterbox_params(w, h, img_size, ih)
        lg = F.interpolate(lg[None, :, py:py + rh, px:px + rw], (h, w), mode="bilinear", align_corners=False)[0]
        return lg.argmax(0).cpu().numpy()

    @torch.no_grad()
    def evaluate(self, model, dataset, args, device):
        model.eval()
        loader = make_loader(dataset, args.batch_size, args.workers, False, args.seed)
        conf = None
        t0 = time.time()
        bar = Progress(len(loader), f"{args.run_name} | test inference")
        for idx, images, _ in loader:
            logits = model(images.to(device, dtype=torch.float32))
            for k, i in enumerate(idx):
                gt = dataset.labels(i)
                pred = self.to_original(logits[k], args.img_size, gt.shape[1], gt.shape[0], self.img_h)
                conf = semantic_confusion(pred, gt, len(self.class_names), conf)
            bar.update()
        bar.close()
        fps = len(dataset) / max(time.time() - t0, 1e-9)
        iou, miou = miou_from_confusion(conf)
        miou_fg = float(np.nanmean(iou[1:]))
        acc = float(np.diag(conf).sum() / max(conf.sum(), 1))
        split = getattr(args, "split_label", self.spec["test_split"])
        names = ["background"] + self.class_names
        print(f"\n{self.spec['dataset'].upper()} {split} ({len(dataset)} images) | semantic segmentation")
        for n, v in zip(names, iou):
            print(f"{n:<14}{'  -   ' if np.isnan(v) else f'{v:.4f}'}")
        print(f"mIoU (incl. background) {miou:.4f}   mIoU (objects) {miou_fg:.4f}   pixel acc {acc:.4f}")
        print(f"Inference speed incl. resize: {fps:.1f} img/s")
        per = {n: (None if np.isnan(v) else float(v)) for n, v in zip(names, iou)}
        return {"metric": "mIoU (VOC confusion matrix, void ignored)", "images": len(dataset), "fps": fps,
                "miou": {"summary": {"mAP": miou}, "per_class": per}, "miou_fg": {"summary": {"mAP": miou_fg}},
                "pixel_acc": acc}
