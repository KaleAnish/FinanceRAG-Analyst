"""Upload a finished runtime bundle to R2 so the deployed API can fetch it.

    ./.venv/bin/python scripts/publish_runtime.py --version runtime-v2-mvp50

Uses the R2 credentials in .env. Uploads to runtime/<version>/ in your bucket.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_DIR / ".env")

from finance_rag.runtime_remote import publish_runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--builds-dir", type=Path, default=PROJECT_DIR / "artifacts" / "builds")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    prefix = f"runtime/{args.version}"
    manifest = publish_runtime(args.builds_dir / args.version, prefix)
    total = sum(f["bytes"] for f in manifest["files"].values()) / 1e6
    print(f"\nPublished {len(manifest['files'])} files ({total:.0f} MB) to r2://<bucket>/{prefix}/")
    print(f"Set on the server: FINANCE_RAG_RUNTIME_R2_PREFIX={prefix}")


if __name__ == "__main__":
    main()
