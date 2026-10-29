import numpy as np
from hyrax.datasets import HyraxDataset

from hyrax_seldon.collate import LightCurveCollationMixin


class ElasticcParquetDataset(LightCurveCollationMixin, HyraxDataset):
    """ELAsTiCC light curves from a parquet file, one object per row, as raw per-point fields.

    The file has SELDON's ELAsTiCC schema: list columns ``MJD``, ``FLUXCAL``, ``FLUXCALERR``,
    ``BAND`` and ``PHOTFLAG``, and a scalar ``SNID``. Every point is emitted as stored;
    ``HyraxSeldon`` leaves selecting points (dropping saturated ones, and non-detections if
    the run excluded them) to SELDON.

    Fields: ``object_id`` (the ``SNID``, as a string), and per point ``time`` (``MJD``),
    ``flux`` (``FLUXCAL``), ``flux_err`` (``FLUXCALERR``), ``band`` (``BAND``) and ``photflag``
    (``PHOTFLAG``). The per-point fields collate padded to the longest light curve in the
    batch, each with a ``<field>_mask``.
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

    def get_object_id(self, idx):
        """The object's ``SNID``."""
        return str(self.table.iloc[idx]["SNID"])

    def get_time(self, idx):
        """``MJD`` of each point."""
        return np.asarray(self.table.iloc[idx]["MJD"], dtype=np.float64)

    def get_flux(self, idx):
        """``FLUXCAL`` of each point."""
        return np.asarray(self.table.iloc[idx]["FLUXCAL"], dtype=np.float64)

    def get_flux_err(self, idx):
        """``FLUXCALERR`` of each point."""
        return np.asarray(self.table.iloc[idx]["FLUXCALERR"], dtype=np.float64)

    def get_band(self, idx):
        """``BAND`` of each point."""
        return np.asarray(self.table.iloc[idx]["BAND"]).astype(str)

    def get_photflag(self, idx):
        """``PHOTFLAG`` of each point."""
        return np.asarray(self.table.iloc[idx]["PHOTFLAG"], dtype=np.int64)
