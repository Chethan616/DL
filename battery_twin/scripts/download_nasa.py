"""Download the official NASA PCoE battery archive without auto-running it."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen


NASA_BATTERY_ZIP = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and optionally extract NASA PCoE Battery Data Set")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--extract", action="store_true", help="Extract the archive into output-dir")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "5. Battery Data Set.zip"
    if not archive.exists():
        request = Request(NASA_BATTERY_ZIP, headers={"User-Agent": "adaptive-battery-twin/0.1"})
        with urlopen(request, timeout=60) as response, archive.open("wb") as handle:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\rDownloaded {downloaded / total:.0%}", end="", flush=True)
        print(f"\nSaved {archive}")
    else:
        print(f"Archive already exists: {archive}")
    if args.extract:
        import zipfile
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(args.output_dir)
        # NASA distributes the battery collection as nested archives. Promote
        # only the four cells committed to this project into data/raw/ so the
        # loader cannot accidentally train on a different subset.
        wanted = {"B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"}
        promoted = []
        for nested in sorted(args.output_dir.rglob("*.zip")):
            with zipfile.ZipFile(nested) as zf:
                for member in zf.namelist():
                    if Path(member).name in wanted:
                        destination = args.output_dir / Path(member).name
                        if not destination.exists():
                            with zf.open(member) as source, destination.open("wb") as target:
                                target.write(source.read())
                            promoted.append(destination.name)
        print(f"Extracted archive into {args.output_dir.resolve()}")
        print(f"Promoted project cells: {', '.join(promoted) if promoted else 'already present'}")


if __name__ == "__main__":
    main()
