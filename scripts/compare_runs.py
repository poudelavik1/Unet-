"""Compare finished training runs in one set of figures and tables (the optimiser and loss study).

    python scripts/compare_runs.py runs/ablation --output runs/ablation/comparison

Each argument is a run directory, or a folder whose subdirectories are runs. A run is placed
by its loss and coloured by its optimiser, as read from its configuration.json; runs without
test_metrics.json (still training) are left out.
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from unetpp.comparison import compare_runs

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories, or folders of run directories")
    parser.add_argument("--output", type=Path, help="Output folder (default: <first argument>/comparison)")
    parser.add_argument("--mode", choices=("tta", "no_tta"), default="tta",
                        help="Test metrics with flip TTA (default) or without")
    args = parser.parse_args()
    output = args.output or args.runs[0] / "comparison"
    written = compare_runs(args.runs, output, args.mode)
    print(f"Written to {output}: {', '.join(written)}")


if __name__ == "__main__":
    main()
