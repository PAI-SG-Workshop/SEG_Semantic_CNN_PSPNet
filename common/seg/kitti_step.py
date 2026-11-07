"""
KITTI-STEP for the video segmentation stage (vseg01, 2026-10-31).

    labels : datasets/kitti-step/panoptic_maps/{train,val}/<seq>/<frame>.png
             R = semantic class (Cityscapes 19 trainIds, 255 = void), G * 256 + B = instance id (car / person only,
             0 = no instance; consistent over time = track id)
    images : datasets/KITTI_MOTS/training/image_02/<seq>/<frame>.png (the same 21 KITTI tracking sequences)
    splits : trainval = STEP train (12 sequences, 5,027 frames), test = STEP val (9 sequences, 2,981 frames,
             identical to KITTI-MOTS val)

Semantic contract (tasks/semseg_task.py, unchanged models): label = trainId + 1 (1..19), 255 = void and the
letterbox padding. [Unified] index 0 ("background") is kept so every semseg model keeps K = C + 1; it never has
ground truth on KITTI-STEP, so the reported "mIoU (objects)" = mean IoU over the 19 classes = the official
KITTI-STEP semantic mIoU (= SQ of STQ when computed over the whole split).

Wide canvas: W x H = img_size x img_h (1216 x 384, the MOT-stage input). KITTI frames are 1224-1242 x 370-376,
scale 0.979-0.993. Augmentation = profile v1.1 (HSV, scale / translate jitter about the centre, hflip), the same
random-number order as VOCSemSeg, applied on the wide canvas (translation relative to W and H).
Frame order: round-robin over sequences, so `max_samples` subsets span every scene (as kittimots_to_voc.py).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from common.datasets.voc import hsv_jitter, letterbox, letterbox_params, to_tensor
from common.seg.step_labels import IGNORE, NUM_CLASSES, SPLIT_DIR, read_panoptic, sequence_frames  # noqa: F401


def letterbox_labels_wide(lab, w_out, h_out):
    h, w = lab.shape
    _, px, py, rw, rh = letterbox_params(w, h, w_out, h_out)
    out = np.full((h_out, w_out), IGNORE, np.uint8)
    out[py:py + rh, px:px + rw] = cv2.resize(lab, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return out


class KittiStepSemSeg(torch.utils.data.Dataset):
    def __init__(self, root, split, img_size, class_names, train=False, hflip_prob=0.0, max_samples=None, aug=None,
                 img_h=384, image_root=None):
        from common import config as C
        self.root, self.img_size, self.img_h = Path(root), int(img_size), int(img_h)
        self.image_root = Path(image_root or C.KITTI_TRACK_IMAGES)
        self.class_names = list(class_names)
        if len(self.class_names) != NUM_CLASSES:
            raise ValueError(f"KITTI-STEP has {NUM_CLASSES} classes, got {len(self.class_names)}")
        self.train, self.aug = train, (aug if train else None)
        self.hflip_prob = hflip_prob if train else 0.0
        self.split_dir = SPLIT_DIR[split]
        seqs = sequence_frames(self.root, split)
        for s, fr in seqs.items():                       # images must exist for the first and last labelled frame
            for f in (fr[0], fr[-1]):
                if not (self.image_root / s / f"{f}.png").is_file():
                    raise FileNotFoundError(self.image_root / s / f"{f}.png")
        lists = [[(s, f) for f in fr] for s, fr in seqs.items()]
        self.ids = []
        for k in range(max(map(len, lists))):
            for lst in lists:
                if k < len(lst):
                    self.ids.append(lst[k])
        if max_samples:
            self.ids = self.ids[:max_samples]
        # [Unified] vseg01b (2026-10-31): STEP train has 5,027 frames (odd). With micro batch 2 and drop_last=False the
        # last batch of every epoch holds 1 image, and the 1x1 global-pooling BN of ASPP (DeepLabv3 / v3+) and of the
        # PSPNet pyramid cannot train on 1 value per channel. The training list repeats its first frame once
        # (5,028 entries, one duplicate per epoch) instead of changing the shared loader.
        if train and len(self.ids) % 2:
            self.ids.append(self.ids[0])
        self.sequences = list(seqs)

    def __len__(self):
        return len(self.ids)

    def image_path(self, i):
        s, f = self.ids[i]
        return self.image_root / s / f"{f}.png"

    def label_path(self, i):
        s, f = self.ids[i]
        return self.root / "panoptic_maps" / self.split_dir / s / f"{f}.png"

    def labels(self, i):
        """trainId + 1 (1..19), 255 = void; original resolution."""
        sem, _ = read_panoptic(self.label_path(i))
        out = np.full(sem.shape, IGNORE, np.uint8)
        valid = sem < NUM_CLASSES
        out[valid] = sem[valid] + 1
        return out

    def __getitem__(self, i):
        W, H = self.img_size, self.img_h
        image = Image.open(self.image_path(i)).convert("RGB")
        lab = letterbox_labels_wide(self.labels(i), W, H)
        image, _, _, _ = letterbox(image, W, H)
        if self.aug:
            arr = hsv_jitter(np.asarray(image), self.aug["hue"], self.aug["saturation"], self.aug["exposure"])
            j = self.aug["jitter"]
            sc = np.random.uniform(1.0 - j, 1.0 + j)               # same draws as VOCSemSeg
            tx, ty = np.random.uniform(-j, j, 2) * np.array([W, H])
            m = np.array([[sc, 0, (1.0 - sc) * W / 2.0 + tx], [0, sc, (1.0 - sc) * H / 2.0 + ty]], dtype=np.float32)
            arr = cv2.warpAffine(arr, m, (W, H), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            lab = cv2.warpAffine(lab, m, (W, H), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
            image = Image.fromarray(arr)
        img = to_tensor(image)
        sem = torch.from_numpy(lab.astype(np.int64))
        if self.hflip_prob and np.random.random() < self.hflip_prob:
            img, sem = torch.flip(img, [-1]), torch.flip(sem, [-1])
        return i, img, {"sem": sem}

    @staticmethod
    def collate_fn(batch):
        idx, images, targets = zip(*batch)
        return list(idx), torch.stack(images, 0), {"sem": torch.stack([t["sem"] for t in targets], 0)}
