"""
Run the three toy encoders through PathReliability and print the main values of
each audit, to confirm that the tool is installed and behaves as described in the
README.

Requires the tool (pip install pathreliability, or a source install) and the
synthetic patches:

    python make_dataset.py
    python verify.py
"""
from pathreliability.adapter import load_adapter
from pathreliability.audit import audit
from pathreliability.embed import build_embeddings

MODELS = ("Reliable", "TextureBlind", "Useless")
ARTEFACT = ("BACK", "DEB")


def main():
    header = ("model", "accuracy", "second dataset", "non-diagnostic", "blur 1 / 2 / 3",
              "gate: second dataset")
    print("{:13} {:>8} {:>15} {:>15} {:>20} {:>21}".format(*header))
    for name in MODELS:
        arrays = build_embeddings(load_adapter(f"models.py:{name}"), "sample_data",
                                  far_dir="sample_data_ood")
        report, _ = audit(arrays, artefact_classes=ARTEFACT, n_boot=50)
        det = report["detection"]
        blur = " / ".join(f"{v['auroc']['maha']:.2f}" for v in det["blur"].values())
        print("{:13} {:8.2f} {:15.2f} {:15.2f} {:>20} {:20.0f}%".format(
            name, report["probe"]["summary"]["accuracy"]["mean"],
            det["far"]["far"]["auroc"]["maha"], det["artefact"]["artefact"]["auroc"]["maha"],
            blur, 100 * report["gate"]["reject_far"]))
    print("\nDetection values are AUROCs of the Mahalanobis score (0.5 is chance); the last "
          "column is the\nshare of the second dataset that the gate rejects.")


if __name__ == "__main__":
    main()
