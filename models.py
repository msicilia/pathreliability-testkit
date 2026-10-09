"""
Toy encoders whose behaviour under the PathReliability audit is known in advance,
for trying the tool without a model of your own. Pure numpy: no torch, no GPU, no
downloads.

Each returns hand-built features, so its audit is deterministic and explainable:

  Reliable      colour, contrast and high-frequency content: accurate, and detects the
                second dataset, the non-diagnostic classes and blur
  TextureBlind  colour of a heavily blurred copy: accurate, but detects mild blur at
                chance, because it has already discarded the detail that blur removes
  Useless       overall brightness only: inaccurate, and at chance on blur and on the
                non-diagnostic classes

Run each through the tool, for example:
  pathreliability audit --adapter models.py:Reliable --data ./sample_data \
      --far-data ./sample_data_ood --artefact-classes BACK DEB --out ./out
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from pathreliability.adapter import Encoder

_IDENTITY = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])   # tool passes raw [0,1] images


def _colour_moments(x):                 # x: [N,H,W,3] in [0,1]
    return np.concatenate([x.mean((1, 2)), x.std((1, 2))], axis=1)   # [N,6]


def _high_freq(x):                      # per-channel high-frequency energy -> [N,3]
    out = np.empty((len(x), 3), np.float32)
    for i in range(len(x)):
        for c in range(3):
            out[i, c] = np.abs(x[i, :, :, c] - gaussian_filter(x[i, :, :, c], 1.5)).mean()
    return out


class Reliable(Encoder):
    """Reads colour, contrast and texture: separates the classes, the non-diagnostic
    classes and the second dataset, and notices blur."""
    embed_dim = 9
    normalisation = _IDENTITY

    def load(self):
        pass

    def embed(self, batch):
        x = np.asarray(batch, np.float32)
        return np.concatenate([_colour_moments(x), _high_freq(x)], axis=1).astype(np.float32)


class TextureBlind(Encoder):
    """Discards high-frequency detail: its features are the colour moments of a heavily
    blurred copy. It still separates the differently coloured classes, but it detects
    mild blur of its input at chance."""
    embed_dim = 6
    normalisation = _IDENTITY

    def load(self):
        pass

    def embed(self, batch):
        x = np.asarray(batch, np.float32)
        xb = np.empty_like(x)
        for i in range(len(x)):
            for c in range(3):
                xb[i, :, :, c] = gaussian_filter(x[i, :, :, c], 6)   # discard texture
        return _colour_moments(xb).astype(np.float32)                # [N,6]


class Useless(Encoder):
    """Overall brightness, repeated: carries almost no information about the input.
    Inaccurate, and at chance on blur and on the non-diagnostic classes."""
    embed_dim = 4
    normalisation = _IDENTITY

    def load(self):
        pass

    def embed(self, batch):
        x = np.asarray(batch, np.float32)
        b = x.mean(axis=(1, 2, 3))                            # overall brightness [N]
        rng = np.random.default_rng(0)
        feat = np.repeat(b[:, None], 4, axis=1) + rng.normal(0, 1e-3, (len(x), 4))
        return feat.astype(np.float32)                       # [N,4], ~constant
