"""Command line entry point: ``python -m mojoscf.build [--force]``."""
from __future__ import annotations

import argparse
import sys

from ._backend import build_extension


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compile the mojoscf Mojo kernels.")
    parser.add_argument("--force", action="store_true", help="rebuild even if up to date")
    args = parser.parse_args(argv)
    path = build_extension(force=args.force, verbose=True)
    print(f"[mojoscf] extension ready: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
