"""Build the GitHub Pages site into a static folder.

    python scripts/build_site.py [OUT_DIR]   # default: _site/

Copies site/ and the stage assets from frontend/static/, and renders the
agent replay (frontend/agent_viz/) with the recorded runs baked in. Used by
.github/workflows/pages.yml; standard library only.
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)

    assets = out / "assets"
    assets.mkdir()
    for name in ("background.js", "background.css"):
        shutil.copy(FRONTEND / "static" / name, assets / name)

    viz = FRONTEND / "agent_viz"
    page = (viz / "index.html").read_text()
    page = page.replace("{{STYLE}}", (viz / "style.css").read_text())
    page = page.replace("{{SCRIPT}}", (viz / "script.js").read_text())
    runs = json.loads((FRONTEND / "demo_runs.json").read_text())
    config = json.dumps({"mode": "loop", "runs": runs}).replace("</", "<\\/")
    (out / "replay.html").write_text(page.replace("{{CONFIG}}", config))

    print(f"Built {out}")


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "_site")
