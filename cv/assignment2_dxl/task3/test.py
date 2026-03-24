#!/usr/bin/env python
from pathlib import Path
import runpy


def main() -> None:
    src_path = Path(__file__).resolve().parents[2] / "assignment2" / "task3" / "test.py"
    runpy.run_path(str(src_path), run_name="__main__")


if __name__ == "__main__":
    main()
