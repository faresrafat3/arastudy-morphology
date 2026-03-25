"""Download and extract Arabic Wikipedia dump."""

from __future__ import annotations

import subprocess
from pathlib import Path


WIKI_DUMP_URL = "https://dumps.wikimedia.org/arwiki/latest/arwiki-latest-pages-articles.xml.bz2"


def download_wikipedia(output_dir: str = "data/raw") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dump_file = out / "arwiki-latest-pages-articles.xml.bz2"
    if dump_file.exists():
        return dump_file

    subprocess.run(["wget", "-O", str(dump_file), WIKI_DUMP_URL], check=True)
    return dump_file


def extract_text(dump_file: Path, output_dir: str = "data/raw") -> Path:
    out = Path(output_dir) / "extracted"
    if out.exists():
        return out

    subprocess.run(
        [
            "python",
            "-m",
            "wikiextractor.WikiExtractor",
            str(dump_file),
            "-o",
            str(out),
            "--no-templates",
            "--min_text_length",
            "50",
            "-q",
        ],
        check=True,
    )
    return out
