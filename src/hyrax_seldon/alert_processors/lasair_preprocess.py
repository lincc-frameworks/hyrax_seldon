def prefilter(self, batch):
    """Prefilter the batch of data so that only light curves with sufficient
    observations are retained.

    Parameters
    ----------
    self : object
        The instance of the class calling this function.
    batch : list
        A batch of data to be pre-filtered.

    Returns
    -------
    list
        The pre-filtered batch of data, unchanged from the input.
    """
    min_observations = 32

    return [
        alert
        for alert in batch
        if alert.get("alert") is not None
        and len(alert["alert"].get("diaForcedSourcesList", [])) >= min_observations
    ]


def preprocess(self, batch):
    """Add photometric flags as `photFlag` to each source in the batch.

    Parameters
    ----------
    self : object
        The instance of the class calling this function.
    batch : list
        A batch of data to be pre-processed.

    Returns
    -------
    list
        The pre-processed batch of data, with photometric flags added.
    """

    for sample in batch:
        alert = sample.get("alert")
        if alert is not None:
            for dia_source in alert.get("diaSourcesList", []):
                dia_source["photFlag"] = _calculate_phot_flag(dia_source["psfFlux"], dia_source["psfFluxErr"])
            for dia_forced_source in alert.get("diaForcedSourcesList", []):
                dia_forced_source["photFlag"] = _calculate_phot_flag(
                    dia_forced_source["psfFlux"], dia_forced_source["psfFluxErr"]
                )
    return batch


def _calculate_phot_flag(flux, flux_err):
    return 4096 if flux / flux_err > 5 else 0
