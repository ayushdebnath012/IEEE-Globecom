"""Real-only image sourcing for the five-class corpus.

The pre-audit pipeline listed ``Tawsifur/COVID-CXR-image-classification`` as an
image source, but that identifier does not exist on the HuggingFace Hub -- the
dataset is published on Kaggle. ``load_hf_medical_images`` swallowed the
resulting exception and ``load_medical_image_data`` silently topped the class up
with ``generate_condition_specific_images``, so every COVID-19 example in the
reported runs was a procedurally drawn intensity pattern.

This module supplies the class from the real Kaggle collection and makes the
synthetic path unreachable. Any shortfall now raises instead of being filled, so
a dead source can never again be mistaken for data.

Expected layout after ``kagglehub.dataset_download`` (or a manual unzip) of
``tawsifurrahman/covid19-radiography-database``::

    <root>/COVID-19_Radiography_Dataset/COVID/images/*.png
    <root>/COVID-19_Radiography_Dataset/Normal/images/*.png
    <root>/COVID-19_Radiography_Dataset/Viral Pneumonia/images/*.png
    <root>/COVID-19_Radiography_Dataset/Lung_Opacity/images/*.png

``Lung_Opacity`` is deliberately unmapped. The pre-audit registry aliased it to
COVID19, which conflates a non-specific finding with a PCR-confirmed diagnosis.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Folder name -> our condition label. Kept narrow on purpose.
KAGGLE_CLASS_DIRS: Dict[str, str] = {
    "COVID": "COVID19",
    "Normal": "NORMAL",
    "Viral Pneumonia": "PNEUMONIA",
}

PROTOCOL = "controlled_v3_real5_template_text"


def _resolve_root(root: Path) -> Path:
    """Accept either the download root or the dataset folder inside it."""
    if (root / "COVID").is_dir():
        return root
    inner = root / "COVID-19_Radiography_Dataset"
    if (inner / "COVID").is_dir():
        return inner
    for child in sorted(p for p in root.glob("*") if p.is_dir()):
        if (child / "COVID").is_dir():
            return child
    raise FileNotFoundError(
        f"no COVID-19_Radiography_Dataset layout under {root}; expected a "
        f"'COVID' folder (got {[p.name for p in root.glob('*')][:8]})")


def _class_files(class_dir: Path) -> List[Path]:
    """Images for one class, sorted so selection is reproducible."""
    images = class_dir / "images"
    search = images if images.is_dir() else class_dir
    files = sorted(p for p in search.rglob("*")
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return files


def load_kaggle_radiography_with(mf, root, wanted: Dict[str, int],
                                 img_size: int = 224, seed: int = 0):
    """Load ``wanted`` real radiographs per condition label from the Kaggle set.

    Returns tensors shaped like the HuggingFace loader's output (ImageNet
    normalized, half precision) with ``[class_index]`` labels.
    """
    import torchvision.transforms as T
    from PIL import Image

    base = _resolve_root(Path(root))
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rng = random.Random(seed)
    images: List = []
    labels: List = []
    picked: Dict[str, int] = {}

    for folder, label in KAGGLE_CLASS_DIRS.items():
        need = wanted.get(label, 0)
        if need <= 0:
            continue
        class_dir = base / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class folder {class_dir}")
        files = _class_files(class_dir)
        if len(files) < need:
            raise RuntimeError(
                f"{label}: {len(files)} real images available under "
                f"{class_dir}, {need} required -- refusing to pad")
        chosen = rng.sample(files, need)
        idx = mf.LABEL_TO_IDX[label]
        loaded = 0
        for path in chosen:
            try:
                with Image.open(path) as im:
                    tensor = transform(im.convert("RGB")).half()
            except Exception:
                continue
            images.append(tensor)
            labels.append([idx])
            loaded += 1
        if loaded < need:
            raise RuntimeError(
                f"{label}: only {loaded}/{need} images decoded from {class_dir}")
        picked[label] = loaded

    print(f"  [kaggle] real images loaded: {picked} from {base}")
    return images, labels


def install_real_only_sources(mf, covid_root, seed: int = 0):
    """Replace the image path so every class comes from a real collection.

    Order of supply per class:

    1. the HuggingFace sources already patched in (keremberke, NIH),
    2. the Kaggle Radiography Database for whatever is still short,
    3. nothing -- a remaining shortfall raises.

    The synthetic generators are replaced with raisers so no later edit can
    reintroduce a silent fallback.
    """
    n_classes = len(mf.CONDITION_LABELS)
    original_hf = mf.load_hf_medical_images

    def load_medical_image_data(n_per_class: int = 200, img_size: int = 224):
        all_images: List = []
        all_labels: List = []

        for _key, cfg in mf.MEDICAL_IMAGE_DATASETS.items():
            if cfg["name"].startswith("Tawsifur/"):
                # Not on the Hub. Supplied from the Kaggle folder below.
                continue
            imgs, lbls = original_hf(cfg["name"], cfg["condition_mapping"],
                                     n_per_class, img_size)
            all_images.extend(imgs)
            all_labels.extend(lbls)
            counts = Counter(l[0] for l in all_labels)
            if all(counts.get(i, 0) >= n_per_class for i in range(n_classes)):
                break

        counts = Counter(l[0] for l in all_labels)
        wanted = {}
        for idx in range(n_classes):
            need = n_per_class - counts.get(idx, 0)
            if need > 0:
                wanted[mf.IDX_TO_LABEL[idx]] = need

        if wanted:
            imgs, lbls = load_kaggle_radiography_with(
                mf, covid_root, wanted, img_size=img_size, seed=seed)
            all_images.extend(imgs)
            all_labels.extend(lbls)

        counts = Counter(l[0] for l in all_labels)
        short = {mf.IDX_TO_LABEL[i]: counts.get(i, 0)
                 for i in range(n_classes) if counts.get(i, 0) < n_per_class}
        if short:
            raise RuntimeError(
                f"real-only sourcing incomplete, need {n_per_class}/class, "
                f"short: {short}. No synthetic substitution is performed.")

        combined = list(zip(all_images, all_labels))
        random.shuffle(combined)
        all_images, all_labels = zip(*combined) if combined else ([], [])
        print(f"  [real-only] {len(all_images)} images: "
              f"{dict(Counter(l[0] for l in all_labels))}")
        return list(all_images), list(all_labels)

    def _refuse(*_a, **_k):
        raise RuntimeError(
            "synthetic image generation is disabled under the real-only "
            "protocol (real_covid_loader.install_real_only_sources)")

    mf.load_medical_image_data = load_medical_image_data
    mf.generate_condition_specific_images = _refuse
    mf.generate_synthetic_image_data = _refuse
    print(f"  [patch] real-only image sourcing installed (covid_root={covid_root})")
    return True
