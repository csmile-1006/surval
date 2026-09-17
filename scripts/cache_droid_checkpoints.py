"""Produce DROID policy caches using the installed training dependency."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    from robomimic.scripts.sequential_cache_checkpoints import main
    main()
