"""
Toy encoders with KNOWN reliability verdicts, for exercising PathReliability when
you don't yet have your own model. Pure numpy — no torch, no GPU, no downloads.

Each subclasses the tool's Encoder and returns hand-built features, so its
behaviour on the audit is deterministic and explainable:

  Reliable      colour + texture + high-frequency features   -> PASS  (a healthy encoder)
  TextureBlind  global colour only                           -> FAIL  (blind to blur AND
                                                                        to background/debris,
                                                                        which share tissue colour)
  Useless       a near-constant statistic                    -> FAIL  (no novelty separation)

Run against the tool, e.g.:
  pathreliability audit --adapter models.py:Reliable     --data ./sample_data --ood-data ./sample_data_ood
  pathreliability audit --adapter models.py:TextureBlind --data ./sample_data --ood-data ./sample_data_ood
  pathreliability audit --adapter models.py:Useless      --data ./sample_data --ood-data ./sample_data_ood
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
    """Reads colour, contrast and texture -> separates classes, rejects artefacts,
    and notices blur. Expected verdict: PASS (maybe WARN on near-OOD)."""
    embed_dim = 9
    normalisation = _IDENTITY

    def load(self):
        pass

    def embed(self, batch):
        x = np.asarray(batch, np.float32)
        return np.concatenate([_colour_moments(x), _high_freq(x)], axis=1).astype(np.float32)


class TextureBlind(Encoder):
    """Throws away high-frequency detail (features are colour moments of a HEAVILY
    pre-blurred copy). It still separates the differently-coloured tissue classes, so
    it classifies — but because it has already discarded texture, it cannot notice
    input blur and cannot tell smooth background / noisy debris from real tissue.
    Expected verdict: FAIL (artifact_rejection_fail; blur undetectable)."""
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
    """A near-constant statistic: carries almost no information about the input.
    Expected verdict: FAIL (far-OOD ~chance; low accuracy)."""
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
