from hyrax.datasets.nested_pandas_dataset import NestedPandasDataset

from hyrax_seldon.datasets.collate import LIGHT_CURVE_FIELDS, LightCurveCollationMixin


class ElasticcParquetDataset(LightCurveCollationMixin, NestedPandasDataset):
    """ELAsTiCC light curves from a parquet file, one object per row, as raw per-point fields.

    The file has SELDON's ELAsTiCC schema: list columns ``MJD``, ``FLUXCAL``, ``FLUXCALERR``,
    ``BAND`` and ``PHOTFLAG``, and a scalar ``SNID``. Every point is emitted as stored;
    ``HyraxSeldon`` leaves selecting points (dropping saturated ones, and non-detections if
    the run excluded them) to SELDON.

    Only those columns are read, and each is a field under its own name; ``SNID`` is the one
    to use as the primary id. The per-point fields collate padded to the longest light curve
    in the batch, each with a ``<field>_mask``.

    Reading is :class:`~hyrax.datasets.nested_pandas_dataset.NestedPandasDataset`'s, so
    ``[data_set.NestedPandasDataset.read_kwargs]`` applies.
    """

    #: The parquet columns read.
    COLUMNS = ["SNID", *LIGHT_CURVE_FIELDS]

    def _load_nested_frame(self, read_kwargs: dict):
        return super()._load_nested_frame({**read_kwargs, "columns": self.COLUMNS})
