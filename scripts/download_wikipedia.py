"""CLI for downloading and extracting Arabic Wikipedia."""

from __future__ import annotations

from src.data.download import download_wikipedia, extract_text


def main() -> None:
    dump = download_wikipedia()
    extracted = extract_text(dump)
    print(f"[ok] dump: {dump}")
    print(f"[ok] extracted: {extracted}")


if __name__ == "__main__":
    main()
