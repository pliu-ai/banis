import os
import sys
import argparse
# Add parent directory to path to import BANIS and inference modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from BANIS import BANIS

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Generate all affinities for BANIS")
    args.add_argument("--checkpoint_path", type=str, default=None,)
    args.add_argument("--prediction_channels", type=int, default=3,)

    args = args.parse_args()

    model = BANIS.load_from_checkpoint(args.checkpoint_path)
    model.full_cube_inference("test", all_seeds=True, evaluate_thresholds=False, prediction_channels=args.prediction_channels)
    # model.full_cube_inference("train", all_seeds=True, evaluate_thresholds=False)
