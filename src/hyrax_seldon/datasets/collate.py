"""Batch collation for datasets that emit SELDON's raw per-point light-curve fields."""

import numpy as np

#: The per-point fields hyrax_seldon's datasets emit, with the value padding uses. They are
#: the list columns of SELDON's ELAsTiCC parquet files, in those files' terms.
LIGHT_CURVE_FIELDS = {
    "MJD": 0.0,
    "FLUXCAL": 0.0,  # ELAsTiCC FLUXCAL scale (zero point 27.5)
    "FLUXCALERR": 0.0,
    "BAND": "",  # band name, e.g. "u".."y"
    "PHOTFLAG": 0,  # ELAsTiCC PHOTFLAG: 4096 marks a detection, 1024 saturation
}


def collate_light_curve_field(samples: list[dict], field: str) -> dict[str, np.ndarray]:
    """Pad one per-point field across a batch to the longest light curve.

    Parameters
    ----------
    samples : list[dict]
        One dict of fields per sample.
    field : str
        The field to collate.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``<field>``, ``(batch, max_length)`` padded values, and ``<field>_mask``,
        ``(batch, max_length)`` bool, true where there is a real point.
    """
    values = [np.asarray(sample[field]) for sample in samples]
    length = max((len(value) for value in values), default=0)
    dtype = np.result_type(*values) if values else np.float64

    padded = np.full((len(values), length), LIGHT_CURVE_FIELDS[field], dtype=dtype)
    mask = np.zeros((len(values), length), dtype=bool)
    for row, value in enumerate(values):
        padded[row, : len(value)] = value
        mask[row, : len(value)] = True
    return {field: padded, f"{field}_mask": mask}


class LightCurveCollationMixin:
    """``collate_<field>`` methods padding each per-point light-curve field, with a mask."""

    def collate_MJD(self, samples):  # noqa: N802
        """Pad ``MJD`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "MJD")

    def collate_FLUXCAL(self, samples):  # noqa: N802
        """Pad ``FLUXCAL`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "FLUXCAL")

    def collate_FLUXCALERR(self, samples):  # noqa: N802
        """Pad ``FLUXCALERR`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "FLUXCALERR")

    def collate_BAND(self, samples):  # noqa: N802
        """Pad ``BAND`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "BAND")

    def collate_PHOTFLAG(self, samples):  # noqa: N802
        """Pad ``PHOTFLAG`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "PHOTFLAG")
