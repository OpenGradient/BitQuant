"""Import smoke test.

Imports every non-test module in the core packages to catch breakage from
dependency upgrades (e.g. an opengradient SDK bump removing an attribute we
use at import time). Runs with dummy credentials -- see .github/workflows/ci.yml.
"""

import importlib
import os
import pkgutil
import sys

# Make the repo root importable when running as `python scripts/import_smoke_test.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PACKAGES = ["agent", "onchain", "api"]

# server.* is excluded from the package walk because server/firebase.py
# initializes Firebase at import time and requires real credentials.


def is_test_module(name: str) -> bool:
    return any(part.startswith("test_") for part in name.split("."))


def main() -> int:
    failed = []
    imported = 0

    for pkg_name in PACKAGES:
        pkg = importlib.import_module(pkg_name)
        modules = [pkg_name] + [
            m.name for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + ".")
        ]
        for name in modules:
            if is_test_module(name):
                continue
            try:
                importlib.import_module(name)
                imported += 1
            except Exception as e:  # noqa: BLE001 - report every failure
                failed.append((name, e))

    print(f"Imported {imported} modules successfully.")
    if failed:
        for name, e in failed:
            print(f"FAILED: {name}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
