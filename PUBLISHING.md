# Publishing the testkit to GitHub

```bash
cd pathreliability-testkit
git init && git add . && git commit -m "PathReliability testkit — toy models with known verdicts"
gh repo create pathreliability-testkit --public --source=. --push \
  --description "Toy encoders with known reliability verdicts to exercise PathReliability"
```

Generated data (`sample_data/`, `sample_data_ood/`) and `out/` are git-ignored — they are
recreated by `python make_dataset.py`. Link this repo from the main tool's README.
