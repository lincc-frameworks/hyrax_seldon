# ruff: noqa: N802, N806

import numpy as np
from hyrax.datasets.collate_utils import collate_as_1d_light_curve
from hyrax_alerts.consumers.lasair_lite_lc_consumer import LasairLiteConsumer
from hyrax_alerts.logging_utils import get_logger

logger = get_logger(__name__)


class LasairSeldonConsumer(LasairLiteConsumer):
    """A subclass of the LasairLiteConsumer that implements a getter and collation
    function for the `photFlag` field that is added to the light curve data during
    preprocessing.
    """

    def __init__(self, config, data_location=None):
        """Initialize the Seldon consumer.

        Parameters
        ----------
        config : dict
            The configuration dictionary for the consumer.
        data_location : str, optional
            The location of the data stream. Defaults to None.
        """
        super().__init__(config=config, data_location=data_location)

    def get_diaForcedSource_photFlag(self, sample):
        """Get the photometric flag from a sample."""
        return self._get_light_curve_values(sample, "diaForcedSourcesList", "photFlag", np.float32)

    def collate_diaForcedSource_photFlag(self, samples):
        """Collate photometric flag data from a list of samples. Returns a dictionary with
        the collated photFlag and a corresponding photFlag_mask indicating valid values."""
        return collate_as_1d_light_curve(samples, "diaForcedSource_photFlag")
