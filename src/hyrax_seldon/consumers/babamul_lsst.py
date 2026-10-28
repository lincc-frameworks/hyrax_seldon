"""Babamul LSST alerts as SELDON's raw per-point light-curve fields.

Babamul LSST alerts (Kafka topics ``babamul.lsst.*``) carry the object's accumulated
history: ``candidate`` (the alert's diaSource), ``prv_candidates`` (DIA detections, which
include the alert's own) and ``fp_hists`` (forced photometry). Times are Julian dates, and
``psfFlux`` / ``psfFluxErr`` are in nJy.
"""

import numpy as np
from hyrax_alerts.consumers.babamul_consumer import BabamulConsumer

from hyrax_seldon.collate import LightCurveCollationMixin

#: JD - MJD.
MJD_OFFSET = 2400000.5

#: nJy (AB zero point 31.4) to ELAsTiCC FLUXCAL (zero point 27.5).
NJY_TO_FLUXCAL = 10 ** (-0.4 * (31.4 - 27.5))

#: The ELAsTiCC ``PHOTFLAG`` bit SELDON reads as a detection.
DETECTION_PHOTFLAG = 4096


def merge_photometry(alert: dict) -> dict[str, np.ndarray]:
    """Merge an alert's detections and forced photometry into one light curve.

    Points from ``prv_candidates`` (detections) and ``fp_hists`` (forced photometry) are
    combined and deduplicated on ``(jd, band)``, with the detection kept when both exist.
    Points without a flux or flux error, and points later than ``candidate.jd``, are dropped.

    Parameters
    ----------
    alert : dict
        A decoded Babamul LSST alert.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``jd``, ``flux`` and ``flux_err`` (nJy), ``band`` and ``detected``, sorted by
        ``(jd, band)``.
    """
    alert_jd = alert["candidate"]["jd"]
    points = {}
    # Forced photometry first, so a detection at the same (jd, band) replaces it.
    for detected, history in (
        (False, alert.get("fp_hists") or []),
        (True, alert.get("prv_candidates") or []),
    ):
        for point in history:
            if point["psfFlux"] is None or point["psfFluxErr"] is None or point["jd"] > alert_jd:
                continue
            points[(point["jd"], point["band"])] = (point["psfFlux"], point["psfFluxErr"], detected)

    keys = sorted(points)
    values = [points[key] for key in keys]
    return {
        "jd": np.array([jd for jd, _ in keys], dtype=np.float64),
        "band": np.array([band for _, band in keys], dtype=str),
        "flux": np.array([flux for flux, _, _ in values], dtype=np.float64),
        "flux_err": np.array([flux_err for _, flux_err, _ in values], dtype=np.float64),
        "detected": np.array([detected for _, _, detected in values], dtype=bool),
    }


class BabamulLsstConsumer(LightCurveCollationMixin, BabamulConsumer):
    """Babamul LSST alerts as raw per-point light curves, for ``HyraxSeldon``.

    Fields: ``candid`` (the primary id), ``objectId``, and per point ``time`` (MJD),
    ``flux`` and ``flux_err`` (nJy converted to the ELAsTiCC ``FLUXCAL`` scale), ``band``
    (``u``..``y``) and ``photflag`` (ELAsTiCC ``PHOTFLAG``: :data:`DETECTION_PHOTFLAG` for DIA
    detections, 0 for forced-only points). See :func:`merge_photometry` for how the light
    curve is assembled. The per-point fields collate padded to the longest light curve in the
    batch, each with a ``<field>_mask``.

    Subscribe to every Babamul LSST topic with
    ``[data_set.KafkaStreamDataset] topics = "^babamul\\\\.lsst\\\\..*"``. ZTF matches in
    ``survey_matches`` are not used.
    """

    def get_objectId(self, msg):  # noqa: N802
        """The LSST object id."""
        return msg["objectId"]

    def get_time(self, msg):
        """MJD of each point."""
        return merge_photometry(msg)["jd"] - MJD_OFFSET

    def get_flux(self, msg):
        """``psfFlux`` of each point, on the ELAsTiCC ``FLUXCAL`` scale."""
        return merge_photometry(msg)["flux"] * NJY_TO_FLUXCAL

    def get_flux_err(self, msg):
        """``psfFluxErr`` of each point, on the ELAsTiCC ``FLUXCAL`` scale."""
        return merge_photometry(msg)["flux_err"] * NJY_TO_FLUXCAL

    def get_band(self, msg):
        """Band of each point."""
        return merge_photometry(msg)["band"]

    def get_photflag(self, msg):
        """ELAsTiCC ``PHOTFLAG`` of each point: a detection for DIA detections, 0 for forced-only points."""
        return np.where(merge_photometry(msg)["detected"], DETECTION_PHOTFLAG, 0).astype(np.int64)
