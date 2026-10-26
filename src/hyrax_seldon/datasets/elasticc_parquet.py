import numpy as np
from hyrax.datasets import HyraxDataset

from hyrax_seldon.collate import LightCurveCollationMixin

#: ELAsTiCC PHOTFLAG bit for a saturated point.
SATURATION_FLAG = 1024


class ElasticcParquetDataset(LightCurveCollationMixin, HyraxDataset):
    """ELAsTiCC light curves from a parquet file, one object per row, as raw per-point fields.

    The file has SELDON's ELAsTiCC schema: list columns ``MJD``, ``FLUXCAL``, ``FLUXCALERR``,
    ``BAND`` and ``PHOTFLAG``, and a scalar ``SNID``. Saturated points (``PHOTFLAG & 1024``)
    are dropped, as SELDON does; non-detections are kept and flagged.

    Fields: ``object_id`` (the ``SNID``, as a string), and per point ``time`` (MJD), ``flux``
    and ``flux_err`` (``FLUXCAL`` scale), ``band`` and ``detected`` (``PHOTFLAG > 0``). The
    per-point fields collate padded to the longest light curve in the batch, each with a
    ``<field>_mask``.
    """

    COLUMNS = ["SNID", "MJD", "FLUXCAL", "FLUXCALERR", "BAND", "PHOTFLAG"]

    def __init__(self, config: dict, data_location=None):
        import pandas as pd

        if data_location is None:
            raise ValueError("ElasticcParquetDataset needs a data_location: an ELAsTiCC parquet file.")

        self.data_location = str(data_location)
        read_kwargs = config["data_set"]["ElasticcParquetDataset"]["read_kwargs"]
        self.table = pd.read_parquet(self.data_location, columns=self.COLUMNS, **read_kwargs)
        super().__init__(config)

    def __len__(self):
        return len(self.table)

    def _kept(self, idx):
        """Return the row and the mask of its non-saturated points."""
        row = self.table.iloc[idx]
        photflag = np.asarray(row["PHOTFLAG"], dtype=np.int64)
        return row, (photflag & SATURATION_FLAG) == 0

    def get_object_id(self, idx):
        """The object's ``SNID``."""
        return str(self.table.iloc[idx]["SNID"])

    def get_time(self, idx):
        """MJD of each point."""
        row, kept = self._kept(idx)
        return np.asarray(row["MJD"], dtype=np.float64)[kept]

    def get_flux(self, idx):
        """``FLUXCAL`` of each point."""
        row, kept = self._kept(idx)
        return np.asarray(row["FLUXCAL"], dtype=np.float64)[kept]

    def get_flux_err(self, idx):
        """``FLUXCALERR`` of each point."""
        row, kept = self._kept(idx)
        return np.asarray(row["FLUXCALERR"], dtype=np.float64)[kept]

    def get_band(self, idx):
        """Band name of each point."""
        row, kept = self._kept(idx)
        return np.asarray(row["BAND"]).astype(str)[kept]

    def get_detected(self, idx):
        """Whether each point is a detection (``PHOTFLAG > 0``)."""
        row, kept = self._kept(idx)
        return (np.asarray(row["PHOTFLAG"], dtype=np.int64) > 0)[kept]
