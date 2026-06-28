"""
Redrob Hackathon — Entry Point
================================
Usage:
    python main.py                                          # uses defaults
    python main.py --candidates candidates.jsonl            # explicit input
    python main.py --candidates candidates.jsonl --out submission.csv
    python main.py --candidates candidates.jsonl --out submission.csv --top-n 100

This file is a thin wrapper. All ranking logic lives in rank.py.
"""

import sys
import time
from pathlib import Path


def main():
    print("=" * 60)
    print("  Redrob AI — Senior AI Engineer Candidate Ranker")
    print("=" * 60)

    # Import the ranker
    try:
        from rank import main as run_ranker
    except ImportError:
        print("\nERROR: rank.py not found in the same folder as main.py.")
        print("Make sure rank.py is in:", Path(__file__).parent)
        sys.exit(1)

    start = time.time()
    print()

    # Run the ranker (it reads --candidates and --out from sys.argv)
    output_path = run_ranker()

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Output: {output_path}")
    print()
    print("Next step: validate your submission")
    print(f"  python validate_submission.py {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()