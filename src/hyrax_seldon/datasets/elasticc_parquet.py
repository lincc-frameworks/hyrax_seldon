from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset

from hyrax_seldon.datasets.collate import LightCurveCollationMixin


class ElasticcParquetDataset(LightCurveCollationMixin, NestedPandasDataset):
    """ELAsTiCC light curves from a parquet file, one object per row, as raw per-point fields.

    The file has SELDON's ELAsTiCC schema: list columns ``MJD``, ``FLUXCAL``, ``FLUXCALERR``,
    ``BAND`` and ``PHOTFLAG``, and a scalar ``SNID``. Every point is emitted as stored;
    ``HyraxSeldon`` leaves selecting points (dropping saturated ones, and non-detections if
    the run excluded them) to SELDON.

    Only those columns are read, renamed to the fields: ``object_id`` (``SNID``), and per point
    ``time`` (``MJD``), ``flux`` (``FLUXCAL``), ``flux_err`` (``FLUXCALERR``), ``band``
    (``BAND``) and ``photflag`` (``PHOTFLAG``). The per-point fields collate padded to the
    longest light curve in the batch, each with a ``<field>_mask``.

    Reading is :class:`~hyrax.datasets.nested_pandas_dataset.NestedPandasDataset`'s, so
    ``[data_set.NestedPandasDataset.read_kwargs]`` applies.
    """

    #: Parquet column -> field name.
    COLUMNS = {
        "SNID": "object_id",
        "MJD": "time",
        "FLUXCAL": "flux",
        "FLUXCALERR": "flux_err",
        "BAND": "band",
        "PHOTFLAG": "photflag",
    }

    def _load_nested_frame(self, read_kwargs: dict):
        frame = super()._load_nested_frame({**read_kwargs, "columns": list(self.COLUMNS)})
        return frame.rename(columns=self.COLUMNS)
