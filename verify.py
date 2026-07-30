"""
Run all three toy models through the tool and print a summary table, so you (or a
colleague) can confirm the tool behaves as expected.

Prereqs:  pip install -e ../pathreliability     # a source install of the tool
          python make_dataset.py                # writes sample_data/ and sample_data_ood/

Run:      python verify.py
"""
from pathreliability.adapter import load_adapter
from pathreliability.audit import audit

MODELS = ["Reliable", "TextureBlind", "Useless"]


def overall(flags):
    vals = flags.values()
    return "FAIL" if "fail" in vals else "WARN" if "warn" in vals else "PASS"


def main():
    print(f"{'model':13} {'OOD':5} {'corrupt':8} {'overall':8}  key metrics")
    print("-" * 74)
    for cls in MODELS:
        r = audit(load_adapter(f"models.py:{cls}"), "sample_data",
                  ood_dir="sample_data_ood", limit=360)
        f = r["verdict"]["per_axis_flags"]
        o = r["axes"]["ood"]
        blur = list(r["axes"]["corruption"]["blur"].values())
        print(f"{cls:13} {f['ood']:5} {f['corruption']:8} {overall(f):8}  "
              f"far={o.get('far_ood',{}).get('auroc_maha')} "
              f"artifact={o.get('artifact_rejection',{}).get('auroc')} "
              f"blur={blur} acc={r['axes']['selective_prediction']['accuracy']}")
    print("\n(The calibration axis is not informative on this toy data and may WARN for "
          "every model;\n focus on the OOD and corruption axes, which separate the three "
          "cleanly.)")


if __name__ == "__main__":
    main()
