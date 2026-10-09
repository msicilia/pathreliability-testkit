# Releasing

The repository is https://github.com/msicilia/pathreliability-testkit.

The synthetic patches (`sample_data/`, `sample_data_ood/`) and `out/` are ignored; they are
recreated by `python make_dataset.py`. After a release of the tool that changes its values, run
`python verify.py` and update the table in the README from its output.
