"""Batch collation for datasets that emit SELDON's raw per-point light-curve fields."""

import numpy as np

#: The per-point fields hyrax_seldon's datasets emit, with the value padding uses.
LIGHT_CURVE_FIELDS = {
    "time": 0.0,
    "flux": 0.0,
    "flux_err": 0.0,
    "band": "",
    "photflag": 0,
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

    def collate_time(self, samples):
        """Pad ``time`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "time")

    def collate_flux(self, samples):
        """Pad ``flux`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "flux")

    def collate_flux_err(self, samples):
        """Pad ``flux_err`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "flux_err")

    def collate_band(self, samples):
        """Pad ``band`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "band")

    def collate_photflag(self, samples):
        """Pad ``photflag`` across the batch; see :func:`collate_light_curve_field`."""
        return collate_light_curve_field(samples, "photflag")
