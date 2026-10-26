#!/usr/bin/env python
"""Download the ELAsTiCC2 training sample and convert it to parquet for hyrax_seldon.

Steps, each skipped when its output is already complete, so the script can be rerun or
resumed after an interruption:

1. Download ``ELASTICC2_TRAIN_02.tar.bz2`` (7.4 GiB) from the DESC ELAsTiCC portal at
   NERSC. A partial download is resumed.
2. Extract it to ``raw/``: one directory per model class, each holding 40 SNANA
   ``*_HEAD.FITS.gz`` / ``*_PHOT.FITS.gz`` pairs.
3. Convert each class to one parquet file with one row per object: every HEAD column as a
   scalar, every PHOT column as a list (``MJD``, ``BAND``, ``FLUXCAL``, ``FLUXCALERR``,
   ``PHOTFLAG``, ...), and ``ELASTICC_class`` from the directory name. This is the schema
   SELDON's parquet loader reads, made the way SELDON's sources do it (ORACLE's
   ``fits_to_parquet.py`` and ``combine_parquet.py``,
   https://github.com/uiucsn/Astro-ORACLE), and the one
   ``hyrax_seldon.datasets.elasticc_parquet.ElasticcParquetDataset`` reads.

Output layout under ``--data-dir`` (default: ``elasticc2_train/`` next to this script)::

    ELASTICC2_TRAIN_02.tar.bz2
    raw/ELASTICC2_TRAIN_02_<class>/...
    parquet/<class>.parquet                       (all files)
    parquet_<n>_files_per_class/<class>.parquet   (with --files-per-class n)

Point a Hyrax data request at one class file, e.g.
``data_location = ".../parquet/SNIa-SALT3.parquet"``.

Usage::

    python download_elasticc.py [--data-dir DIR] [--files-per-class N] [--workers N]
"""

import argparse
import os
import sys
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits

URL = (
    "https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/"
    "ELASTICC2_TRAINING_SAMPLE_2/ELASTICC2_TRAIN_02.tar.bz2"
)
CLASS_PREFIX = "ELASTICC2_TRAIN_02_"
EXTRACTED_MARKER = ".extracted"
CHUNK = 8 * 1024 * 1024


def log(message):
    """Print a progress message."""
    print(message, flush=True)


def remote_size(url: str) -> int:
    """Return the size in bytes of the file at ``url``."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
        return int(response.headers["Content-Length"])


def download(url: str, destination: Path) -> Path:
    """Download ``url`` to ``destination``, resuming a partial file; skip a complete one."""
    size = remote_size(url)
    have = destination.stat().st_size if destination.exists() else 0
    if have == size:
        log(f"Download: {destination} is complete ({size:,} bytes); skipping.")
        return destination
    if have > size:
        raise RuntimeError(f"{destination} is larger than {url} ({have:,} > {size:,} bytes); remove it.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    log(f"Download: {url} -> {destination} ({have:,} of {size:,} bytes present)")
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(request) as response, open(destination, "ab" if have else "wb") as out:
        if have and response.status != 206:
            raise RuntimeError(f"The server did not resume the download of {url} (HTTP {response.status}).")
        next_report = have
        while chunk := response.read(CHUNK):
            out.write(chunk)
            have += len(chunk)
            if have >= next_report:
                log(f"  {have / size:6.1%} ({have / 2**30:.2f} GiB)")
                next_report += 512 * 2**20

    if have != size:
        raise RuntimeError(f"Downloaded {have:,} of {size:,} bytes from {url}; rerun to resume.")
    return destination


def extract(tarball: Path, raw_dir: Path) -> Path:
    """Extract ``tarball`` into ``raw_dir``; skip if a previous extraction of it completed."""
    marker = raw_dir / EXTRACTED_MARKER
    stamp = f"{tarball.name} {tarball.stat().st_size}"
    if marker.exists() and marker.read_text() == stamp:
        log(f"Extract: {raw_dir} already holds {tarball.name}; skipping.")
        return raw_dir

    log(f"Extract: {tarball} -> {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:bz2") as archive:
        # The "data" filter refuses absolute paths, links out of raw_dir, and special files.
        archive.extractall(raw_dir, filter="data")
    marker.write_text(stamp)
    return raw_dir


def _native(values: np.ndarray) -> np.ndarray:
    """FITS data is big-endian; byte strings are space-padded. Return native values."""
    if values.dtype.kind == "S":
        return np.char.decode(np.char.strip(values), "ascii")
    return values.astype(values.dtype.newbyteorder("="))


def read_snana_pair(head_file: Path, phot_file: Path) -> pa.Table:
    """Read one SNANA HEAD/PHOT pair as one row per object.

    As ``sncosmo.read_snana_fits``: each HEAD row is an object, whose photometry is PHOT rows
    ``PTROBS_MIN - 1`` to ``PTROBS_MAX`` (1-based, inclusive), which skips the separator rows
    between objects.
    """
    head = fits.getdata(head_file, 1, view=np.ndarray)
    phot = fits.getdata(phot_file, 1, view=np.ndarray)

    start = head["PTROBS_MIN"].astype(np.int64) - 1
    stop = head["PTROBS_MAX"].astype(np.int64)
    lengths = stop - start
    rows = np.concatenate([np.arange(a, b) for a, b in zip(start, stop, strict=True)])
    offsets = pa.array(np.concatenate([[0], np.cumsum(lengths)]), type=pa.int64())

    columns = {name: pa.array(_native(head[name])) for name in head.dtype.names}
    columns["SNID"] = pa.array(_native(head["SNID"]).astype(np.int64))
    for name in phot.dtype.names:
        columns[name] = pa.LargeListArray.from_arrays(offsets, pa.array(_native(phot[name][rows])))
    return pa.table(columns)


def convert_class(class_dir: Path, output: Path, files_per_class: int | None) -> int:
    """Convert one class directory to ``output``; skip it if ``output`` exists."""
    if output.exists():
        return pq.read_metadata(output).num_rows

    heads = sorted(class_dir.glob("*_HEAD.FITS.gz"))[:files_per_class]
    phots = [head.with_name(head.name.replace("_HEAD.", "_PHOT.")) for head in heads]
    if not heads or not all(phot.exists() for phot in phots):
        raise FileNotFoundError(f"{class_dir} does not hold matching HEAD/PHOT FITS pairs.")

    table = pa.concat_tables(read_snana_pair(head, phot) for head, phot in zip(heads, phots, strict=True))
    class_name = class_dir.name.removeprefix(CLASS_PREFIX)
    table = table.append_column("ELASTICC_class", pa.array([class_name] * table.num_rows))

    # Write to a temporary name first, so an interrupted run never leaves a partial file
    # that a rerun would take as finished.
    partial = output.with_name(output.name + ".partial")
    pq.write_table(table, partial)
    os.replace(partial, output)
    return table.num_rows


def convert(raw_dir: Path, parquet_dir: Path, files_per_class: int | None, workers: int) -> Path:
    """Convert every class directory under ``raw_dir`` to ``parquet_dir/<class>.parquet``."""
    class_dirs = sorted(path for path in raw_dir.glob(f"{CLASS_PREFIX}*") if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No {CLASS_PREFIX}* class directories in {raw_dir}.")

    parquet_dir.mkdir(parents=True, exist_ok=True)
    outputs = [parquet_dir / f"{d.name.removeprefix(CLASS_PREFIX)}.parquet" for d in class_dirs]
    todo = sum(not output.exists() for output in outputs)
    log(f"Convert: {len(class_dirs)} classes -> {parquet_dir} ({len(class_dirs) - todo} already done)")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        counts = pool.map(convert_class, class_dirs, outputs, [files_per_class] * len(class_dirs))
        for output, count in zip(outputs, counts, strict=True):
            log(f"  {output.name}: {count:,} objects")
    return parquet_dir


def main(argv=None):
    """Download, extract and convert the ELAsTiCC2 training sample."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "elasticc2_train",
        help="Where to put the download, the extracted files and the parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--files-per-class",
        type=int,
        default=None,
        help="Convert only the first N of each class's 40 HEAD/PHOT pairs (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of classes converted in parallel (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    tarball = download(URL, data_dir / Path(URL).name)
    raw_dir = extract(tarball, data_dir / "raw")

    name = "parquet" if args.files_per_class is None else f"parquet_{args.files_per_class}_files_per_class"
    parquet_dir = convert(raw_dir, data_dir / name, args.files_per_class, args.workers)
    log(f"Done: {parquet_dir}")


if __name__ == "__main__":
    sys.exit(main())
