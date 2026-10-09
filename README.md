# PathReliability Testkit

Toy encoders with known behaviour, for trying
[PathReliability](https://github.com/msicilia/pathreliability) without a model or data of your
own. Everything is pure numpy: no torch, no GPU, no downloads.

The kit has three small encoders whose features are built by hand, and a generator of synthetic
patches. It is a way to check that the tool is installed and to see what a report looks like. It
says nothing about real models.

## Setup

Install the tool (its Docker image, or `pip install` from its repository), then:

```bash
git clone https://github.com/msicilia/pathreliability-testkit && cd pathreliability-testkit
python make_dataset.py        # writes ./sample_data and ./sample_data_ood (numpy, pillow)
```

`sample_data` has four "tissue" classes and two non-diagnostic ones (`BACK`, smooth, and `DEB`,
noisy). `sample_data_ood` is the same design with a shifted palette and plays the second dataset.

## Run the tool on a toy encoder

```bash
pathreliability audit --adapter models.py:Reliable --data ./sample_data \
    --far-data ./sample_data_ood --artefact-classes BACK DEB --out ./out
```

or, with the Docker image:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  ghcr.io/msicilia/pathreliability audit --adapter /work/models.py:Reliable \
  --data /work/sample_data --far-data /work/sample_data_ood \
  --artefact-classes BACK DEB --out /work/out
```

Open `out/report.html`, and replace `Reliable` with `TextureBlind` or `Useless` to compare. To
run all three and print their main values (this needs a source or pip install of the tool):

```bash
python verify.py
```

## The three encoders and what to expect

| Encoder | Features | Accuracy | Second dataset | Non-diagnostic classes | Blur, σ = 1 / 2 / 3 |
|---|---|---|---|---|---|
| `Reliable` | colour, contrast, high-frequency content | 1.00 | 1.00 | 1.00 | 0.93 / 1.00 / 1.00 |
| `TextureBlind` | colour of a heavily blurred copy | 1.00 | 1.00 | 1.00 | 0.52 / 0.71 / 0.84 |
| `Useless` | overall brightness only | 0.58 | 1.00 | 0.47 | 0.50 / 0.50 / 0.50 |

The detection columns are AUROCs of the Mahalanobis score, where 0.5 is chance. They are the
output of `verify.py` with version 0.2.0 of the tool.

- `Reliable` separates the classes and detects everything it is shown.
- `TextureBlind` classifies as well, but detects mild blur at chance: it has already discarded
  the detail that blur removes. Two encoders with the same accuracy differ here, which is the
  point of auditing beyond accuracy.
- `Useless` is inaccurate and at chance on blur and on the non-diagnostic classes. It still
  detects the second dataset, because the shifted palette changes the one thing it reads,
  brightness.

The synthetic patches carry no patient or slide ids, so the report notes that the split is at
patch level and skips the tuning check. Calibration and error ranking are of little interest on
data this separable.

## Files

- `make_dataset.py`: generates the two synthetic patch sets
- `models.py`: the three toy encoders
- `verify.py`: runs all three and prints their main values

MIT licence.
