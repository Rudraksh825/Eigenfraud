"""
Unified wrapper to evaluate external AI-generated image detectors.

Outputs CSV: path, label, score  (higher score = more likely fake, label: 0=real 1=fake)

Usage:
    python scripts/eval_external.py --detector cnndetection \
        --data data/raw/cifake/test \
        --weights detectors/CNNDetection/weights/blur_jpg_prob0.5.pth \
        --out results/cnndetection_cifake.csv

Detectors and their default weight paths:
    cnndetection  detectors/CNNDetection/weights/blur_jpg_prob0.5.pth
    freqnet       detectors/FreqNet-DeepfakeDetection/4-classes-freqnet-v2.pth
    npr           detectors/NPR-DeepfakeDetection/NPR.pth
    univfd        detectors/UniversalFakeDetect/pretrained_weights/fc_weights.pth
    fatformer     detectors/FatFormer/pretrained/FatFormer_4class.pth
    bfree         detectors/B-Free/code/weights/BFREE_dino2reg4  (directory)

Data directory layout (one level deep):
    test/
      REAL/  (or real/, 0_real/)  -> label 0
      FAKE/  (or fake/, 1_fake/)  -> label 1

Notes:
    - univfd and fatformer auto-download CLIP ViT-L/14 (~890MB) to ~/.cache/clip/ on first run.
    - bfree requires the weights *directory* (containing config.yaml), not a .pth file.
    - fatformer requires pytorch_wavelets: pip install pytorch_wavelets
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score

REPO_ROOT = Path(__file__).parent.parent
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def collect_images(data_dir: Path):
    """Return list of (path_str, label) from a real/fake directory tree."""
    samples = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if 'real' in name:
            label = 0
        elif 'fake' in name:
            label = 1
        else:
            continue
        for f in subdir.rglob('*'):
            if f.suffix.lower() in IMAGE_EXTS:
                samples.append((str(f), label))
    if not samples:
        raise ValueError(
            f"No real/fake images found under {data_dir}. "
            "Subdirectory names must contain 'real' or 'fake'."
        )
    return samples


class ImageListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), label, path
        except Exception:
            return None, label, path

    @staticmethod
    def collate(batch):
        batch = [b for b in batch if b[0] is not None]
        imgs = torch.stack([b[0] for b in batch])
        labels = [b[1] for b in batch]
        paths = [b[2] for b in batch]
        return imgs, labels, paths


# ---------------------------------------------------------------------------
# Shared transforms
# ---------------------------------------------------------------------------

_IMAGENET_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
_CLIP_NORM = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)

# Used by CNNDetection / FreqNet / NPR / FatFormer
CNN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _IMAGENET_NORM,
])

# Used by UnivFD  (bicubic resize to match CLIP preprocessing)
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _CLIP_NORM,
])

# Used by B-Free  (force 256×256 so variable-resolution Defactify images collate cleanly;
# B-Free's internal Wrapper5crops then crops to 224)
BFREE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    _IMAGENET_NORM,
])


# ---------------------------------------------------------------------------
# Detector loaders
# ---------------------------------------------------------------------------

def _auto_load_state(model, weights: str):
    """Load state dict, handling both direct OrderedDicts and {'model': ...} dicts."""
    state = torch.load(weights, map_location='cpu')
    if isinstance(state, dict) and 'model' in state and not any(
        k.startswith('layer') or k.startswith('conv') or k.startswith('bn') or
        k.startswith('weight') or k.startswith('fc')
        for k in state.keys()
    ):
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)


def load_cnndetection(weights: str):
    sys.path.insert(0, str(REPO_ROOT / 'detectors/CNNDetection'))
    from networks.resnet import resnet50
    model = resnet50(num_classes=1)
    state = torch.load(weights, map_location='cpu')
    model.load_state_dict(state['model'])
    return model, CNN_TRANSFORM


def load_freqnet(weights: str):
    sys.path.insert(0, str(REPO_ROOT / 'detectors/FreqNet-DeepfakeDetection'))
    from networks.freqnet import freqnet
    model = freqnet(num_classes=1)
    state = torch.load(weights, map_location='cpu')
    model.load_state_dict(state)
    return model, CNN_TRANSFORM


def _strip_module_prefix(sd):
    """Remove DataParallel 'module.' key prefix if present."""
    if all(k.startswith('module.') for k in sd.keys()):
        return {k[7:]: v for k, v in sd.items()}
    return sd


def load_npr(weights: str):
    sys.path.insert(0, str(REPO_ROOT / 'detectors/NPR-DeepfakeDetection'))
    from networks.resnet import resnet50
    model = resnet50(num_classes=1)
    state = torch.load(weights, map_location='cpu')
    # NPR.pth wraps weights under 'model' (DDP-saved); model_epoch_last_3090.pth is a raw OrderedDict
    sd = state['model'] if isinstance(state, dict) and 'model' in state else state
    model.load_state_dict(_strip_module_prefix(sd))
    return model, CNN_TRANSFORM


def load_univfd(weights: str):
    sys.path.insert(0, str(REPO_ROOT / 'detectors/UniversalFakeDetect'))
    from models import get_model
    model = get_model('CLIP:ViT-L/14')
    state = torch.load(weights, map_location='cpu')
    model.fc.load_state_dict(state)
    return model, CLIP_TRANSFORM


def load_fatformer(weights: str):
    import argparse as _ap
    sys.path.insert(0, str(REPO_ROOT / 'detectors/FatFormer'))
    from models import build_model
    args = _ap.Namespace(
        backbone='CLIP:ViT-L/14',
        num_classes=2,
        num_vit_adapter=3,
        num_context_embedding=8,
        init_context_embedding='',
        hidden_dim=768,
        clip_vision_width=1024,
        frequency_encoder_layer=2,
        decoder_layer=4,
        num_heads=12,
    )
    model = build_model(args)
    state = torch.load(weights, map_location='cpu')
    model.load_state_dict(state['model'])
    return model, CNN_TRANSFORM


def load_bfree(weights: str):
    """weights is the model directory containing config.yaml."""
    import yaml
    sys.path.insert(0, str(REPO_ROOT / 'detectors/B-Free/code'))
    from networks import get_network, load_weights
    weights_dir = Path(weights)
    with open(weights_dir / 'config.yaml') as f:
        cfg = yaml.safe_load(f)
    model = load_weights(get_network(cfg['arch']), str(weights_dir / cfg['weights_file']))
    return model, BFREE_TRANSFORM


LOADERS = {
    'cnndetection': load_cnndetection,
    'freqnet':      load_freqnet,
    'npr':          load_npr,
    'univfd':       load_univfd,
    'fatformer':    load_fatformer,
    'bfree':        load_bfree,
}

DEFAULT_WEIGHTS = {
    'cnndetection': 'detectors/CNNDetection/weights/blur_jpg_prob0.5.pth',
    'freqnet':      'detectors/FreqNet-DeepfakeDetection/4-classes-freqnet-v2.pth',
    'npr':          'detectors/NPR-DeepfakeDetection/NPR.pth',
    'univfd':       'detectors/UniversalFakeDetect/pretrained_weights/fc_weights.pth',
    'fatformer':    'detectors/FatFormer/pretrained/FatFormer_4class.pth',
    'bfree':        'detectors/B-Free/code/weights/BFREE_dino2reg4',
}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, loader, device, detector):
    all_paths, all_labels, all_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            if detector == 'fatformer':
                # FatFormer returns logits [B, 2]; index 1 = fake
                scores = out.softmax(dim=1)[:, 1].cpu().tolist()
            elif detector == 'bfree':
                out_np = out.cpu().numpy()
                if out_np.shape[1] == 1:
                    scores = out_np[:, 0].tolist()
                else:
                    scores = (out_np[:, 1] - out_np[:, 0]).tolist()
            else:
                # CNNDetection / FreqNet / NPR / UnivFD: scalar logit
                scores = out.sigmoid().flatten().cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(labels)
            all_paths.extend(paths)
    return all_paths, all_labels, all_scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate an external fake-image detector on a real/fake dataset.'
    )
    parser.add_argument('--detector', required=True, choices=list(LOADERS.keys()))
    parser.add_argument('--data', required=True, type=Path,
                        help='Directory with real/ and fake/ subdirs')
    parser.add_argument('--weights', default=None,
                        help='Path to weights file or directory (default: see DEFAULT_WEIGHTS)')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output CSV path (path, label, score)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    weights = args.weights or str(REPO_ROOT / DEFAULT_WEIGHTS[args.detector])

    print(f"Detector : {args.detector}")
    print(f"Weights  : {weights}")
    print(f"Data     : {args.data}")
    print(f"Device   : {args.device}")

    print("Loading model...")
    model, transform = LOADERS[args.detector](weights)
    model = model.to(args.device)
    model.eval()

    samples = collect_images(args.data)
    n_real = sum(1 for _, l in samples if l == 0)
    n_fake = sum(1 for _, l in samples if l == 1)
    print(f"Images   : {len(samples)} ({n_real} real, {n_fake} fake)")

    batch_size = args.batch_size
    if args.detector == 'bfree':
        batch_size = min(batch_size, 8)  # 5-crop internally multiplies batch by 5

    dataset = ImageListDataset(samples, transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=ImageListDataset.collate,
    )

    print("Running inference...")
    paths, labels, scores = run_inference(model, loader, args.device, args.detector)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label', 'score'])
        for p, l, s in zip(paths, labels, scores):
            writer.writerow([p, l, round(s, 9)])

    y_true = np.array(labels)
    y_score = np.array(scores)
    auc = roc_auc_score(y_true, y_score)
    acc = accuracy_score(y_true, y_score > 0.5)
    print(f"\nAUC      : {auc:.4f}")
    print(f"Acc@0.5  : {acc:.4f}")
    print(f"Saved    : {args.out}  ({len(paths)} rows)")


if __name__ == '__main__':
    main()
