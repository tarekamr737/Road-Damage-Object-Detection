from __future__ import annotations

import json
from pathlib import Path

from road_damage.training.runtime import environment_summary


def main() -> int:
    summary = environment_summary()
    output = Path("reports/environment.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
