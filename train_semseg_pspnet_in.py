"""Train semseg pspnet_in (ImageNet backbone, vseg01), then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.semseg.models import make_build

if __name__ == "__main__":
    main("train", task="semseg", version="pspnet_in", build_model=make_build("pspnet_in"))
