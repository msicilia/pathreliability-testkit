"""
Generate a tiny SYNTHETIC ImageFolder dataset so you can exercise the tool with
no downloads. Two datasets are written:

  sample_data/      the in-distribution set (4 tissue classes + BACK + DEB)
  sample_data_ood/  a different-"organ" set (different palette) for the far-OOD axis

Design (so the toy encoders have KNOWN verdicts):
  * tissue classes differ in BOTH colour and stripe texture;
  * BACK (smooth) and DEB (noisy) are coloured LIKE a tissue class, so an encoder
    that only reads global colour cannot tell them from real tissue (artifact-blind),
    while an encoder that reads texture/high-frequency content can.

Run:  python make_dataset.py         # writes ./sample_data and ./sample_data_ood
"""
import os
import numpy as np
from PIL import Image

S = 96                      # image size (the tool resizes to 224)
N_PER_CLASS = 100
RNG = np.random.default_rng(0)

# (name, base RGB, stripe freq, stripe orientation)
TISSUE = [
    ("tissue_a", (0.72, 0.40, 0.52), 6, 0.0),
    ("tissue_b", (0.50, 0.30, 0.62), 9, 0.8),
    ("tissue_c", (0.40, 0.52, 0.70), 12, 1.6),
    ("tissue_d", (0.42, 0.60, 0.42), 8, 2.4),
]
# striped tissue is darkened by the modulation (spatial mean factor ~0.8); make the
# artefacts share tissue_a's ACTUAL mean colour so a colour-only encoder cannot tell
# them apart (that is the deliberate trap for TextureBlind).
_STRIPE_MEAN_FACTOR = 0.8


def _grid():
    xs = np.linspace(0, 1, S)
    x, y = np.meshgrid(xs, xs)
    return x, y


def _striped(base, freq, orient, phase):
    x, y = _grid()
    t = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (x * np.cos(orient) + y * np.sin(orient)) + phase)
    img = np.stack([np.clip(c * (0.6 + 0.4 * t), 0, 1) for c in base], axis=-1)
    return img


def _smooth(mean_colour):                 # BACK: flat colour, tiny noise
    img = np.ones((S, S, 3)) * np.array(mean_colour)
    return np.clip(img + RNG.normal(0, 0.015, (S, S, 3)), 0, 1)


def _noisy(mean_colour):                  # DEB: same mean colour, heavy high-freq noise
    img = np.ones((S, S, 3)) * np.array(mean_colour)
    return np.clip(img + RNG.normal(0, 0.28, (S, S, 3)), 0, 1)


def _save(arr, path):
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def build(out, palette_shift=0.0):
    for name, base, freq, orient in TISSUE:
        base = np.clip(np.array(base) + palette_shift, 0, 1)
        d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        for i in range(N_PER_CLASS):
            bj = np.clip(base + RNG.normal(0, 0.02, 3), 0, 1)   # mild within-class spread
            _save(_striped(tuple(bj), freq, orient, RNG.uniform(0, 6.28)),
                  os.path.join(d, f"{i:03d}.png"))
    base_a = np.clip(np.array(TISSUE[0][1]) + palette_shift, 0, 1)
    amc = tuple(_STRIPE_MEAN_FACTOR * base_a)          # == tissue_a's spatial mean colour
    for name, fn in (("BACK", _smooth), ("DEB", _noisy)):
        d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        for i in range(N_PER_CLASS):
            _save(fn(amc), os.path.join(d, f"{i:03d}.png"))
    print(f"wrote {out}/  ({len(TISSUE)} tissue classes + BACK + DEB, "
          f"{N_PER_CLASS} imgs each)")


if __name__ == "__main__":
    build("sample_data", palette_shift=0.0)
    build("sample_data_ood", palette_shift=0.18)     # a shifted palette = "other organ"
