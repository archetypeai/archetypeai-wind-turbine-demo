"""Precompute the n-shot KNN reference library, once, to disk.

The reference set (healthy: WT09 Jul + WT06 Oct, fault: WT05 Mar outage) is
static, so its Omega embeddings never change. Embedding them at runtime on every
cold start just re-does deterministic work and blocks the first classification.

Run this once after changing any reference window / window size / model:

    python build_library.py

It writes `library.json` (scaler + vectors + a config fingerprint). At runtime
`newton_client._build_library()` loads that file instantly and skips the live
embedding calls; if it's missing or the config changed, it rebuilds live.
"""
import logging
import os

from dotenv import load_dotenv

import newton_client as nc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

if __name__ == "__main__":
    load_dotenv()
    if not os.environ.get("ATAI_API_KEY"):
        raise SystemExit("ATAI_API_KEY is not set (check your .env)")
    path = nc.save_library()
    print(f"Wrote {path}")
