# PathReliability Testkit

**Toy models with known reliability verdicts — try [PathReliability](https://github.com/msicilia/pathreliability)
without your own model or data.**

If you want to see what the tool does before wiring up your own encoder, this gives you three
tiny "encoders" whose reliability behaviour is engineered and predictable, plus a synthetic
dataset generated on the fly. Everything is pure numpy — **no torch, no GPU, no downloads.**

## Setup

These toy models are adapters for the
[PathReliability](https://github.com/msicilia/pathreliability) tool, so you need the tool — its
Docker image, or a source install. Then clone this repo and generate the synthetic data:

```bash
git clone https://github.com/msicilia/pathreliability-testkit && cd pathreliability-testkit
python make_dataset.py               # writes ./sample_data and ./sample_data_ood (needs numpy, pillow)
```

## Run the tool on each toy model

With the tool's Docker image, mounting this folder and an output directory:

```bash
docker run --rm -v "$PWD:/work" -v "$PWD/out:/out" \
  pathreliability audit --adapter /work/models.py:Reliable \
  --data /work/sample_data --ood-data /work/sample_data_ood --out /out
```

Swap `Reliable` for `TextureBlind` or `Useless`. Equivalently, with a source install of the tool
(`pip install -e ../pathreliability`):

```bash
pathreliability audit --adapter models.py:Useless --data ./sample_data --ood-data ./sample_data_ood
```

Open each `out/report.html`. To run all three and print a summary table (this requires the source
install, as it imports the tool):

```bash
python verify.py
```

## The three models and their expected verdicts

Each model returns hand-built features, so its behaviour is deterministic and explainable.

| Model | What it "sees" | far-OOD | artefact | blur | accuracy | OOD axis | corruption axis |
|---|---|---|---|---|---|---|---|
| **Reliable** | colour + contrast + texture | 0.97 | 1.00 | 0.94→1.0 | 1.00 | **PASS** | **PASS** |
| **TextureBlind** | colour of a heavily-blurred copy | 0.97 | 1.00 | 0.57→0.84 | 0.83 | PASS | **WARN** (weak at mild blur) |
| **Useless** | one near-constant number | 0.29 | 0.42 | 0.51 | 0.34 | **FAIL** | WARN |

- **Reliable** — a healthy encoder: it flags other-tissue inputs, rejects background/debris, and
  notices blur. Passes every reliability axis.
- **TextureBlind** — it has thrown away high-frequency detail, so it still classifies (colours
  differ) but is **weak at detecting blur** — the corruption axis warns.
- **Useless** — an uninformative encoder: it cannot separate anything, so far-OOD is near chance
  and artefact rejection fails. **The tool blocks it** (`artifact_rejection_fail`).

> **Note on the calibration axis.** On this tiny synthetic dataset the calibration axis is not
> meaningful (a linear probe on near-perfectly-separable toy data is not well-calibrated), so it
> may WARN for every model, including the healthy one. Judge the toy models by the **OOD and
> corruption** axes, which separate them cleanly. On real data the calibration axis is
> informative — see the method paper's PANDA result.

## Use these on your own data

The models are dataset-agnostic — point them at any ImageFolder (one subfolder per class). Name
some classes `BACK` / `DEB` / `ADI` to activate the artefact-rejection axis. This is a quick way
to sanity-check that the tool is installed and behaving before you wrap your real model.

## What's here

- `make_dataset.py` — generates the synthetic ID and far-OOD datasets
- `models.py` — the three toy encoders (`Reliable`, `TextureBlind`, `Useless`)
- `verify.py` — runs all three and prints the summary table

MIT-licensed.
