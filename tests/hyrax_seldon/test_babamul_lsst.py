import numpy as np
import pytest
from babamul_fixtures import (
    LSST_BANDS,
    MJD_OFFSET,
    decode_alert,
    detection,
    encode_alert,
    forced,
    make_alert,
    make_light_curve_alert,
)
from hyrax_alerts.writers.base_writer import HyraxAlertsBaseWriter

from hyrax_seldon.consumers.babamul_lsst import NJY_TO_FLUXCAL, BabamulLsstConsumer, merge_photometry

FIELDS = ["time", "flux", "flux_err", "band", "detected"]


class FakeMessage:
    """Minimal stand-in for a confluent_kafka Message."""

    def __init__(self, value):
        self._value = value

    def value(self):
        """Return the message payload."""
        return self._value

    def error(self):
        """No error."""
        return None


class FakeKafka:
    """Hands out queued messages, then calls ``on_exhausted``."""

    def __init__(self, messages, on_exhausted):
        self._messages = list(messages)
        self._on_exhausted = on_exhausted

    def poll(self, timeout):
        """Pop the next queued message, or None once the queue is empty."""
        if self._messages:
            return self._messages.pop(0)
        self._on_exhausted()
        return None

    def consume(self, num_messages, timeout):
        """Drain up to num_messages queued messages."""
        drained, self._messages = self._messages[:num_messages], self._messages[num_messages:]
        if not drained:
            self._on_exhausted()
        return drained

    def subscribe(self, topics, **kwargs):
        """No-op."""

    def commit(self, *args, **kwargs):
        """No-op."""

    def close(self):
        """No-op."""


def _alerts():
    """Alerts of different lengths, so each result can be matched to its alert."""
    rng = np.random.default_rng(11)
    return [
        make_light_curve_alert(rng, 1001, "lsst-obj-1", n_visits=80),
        make_light_curve_alert(rng, 1002, "lsst-obj-2", n_visits=120),
        make_light_curve_alert(rng, 1003, "lsst-obj-3", n_visits=50, empty_forced=True),
        make_light_curve_alert(rng, 1004, "lsst-obj-4", n_visits=700),
        make_light_curve_alert(rng, 1005, "lsst-obj-5", n_visits=30, n_null=0),
    ]


class FakeBabamulLsstConsumer(BabamulLsstConsumer):
    """The real consumer, reading Avro-encoded alerts from a fake Kafka consumer."""

    def _make_consumer(self):
        return FakeKafka([FakeMessage(encode_alert(alert)) for alert in _alerts()], on_exhausted=self.stop)


@pytest.fixture
def consumer():
    """A BabamulLsstConsumer configured from hyrax_seldon's defaults."""
    from hyrax import Hyrax

    h = Hyrax(setup_logging=False)
    h.set_config("model.name", "hyrax_seldon.models.seldon.HyraxSeldon")
    h.set_config("data_set.KafkaStreamDataset.topics", "babamul.lsst.no-ztf-match.hosted")
    return BabamulLsstConsumer(h.config)


def _small_alert():
    """An alert whose merged light curve is easy to write down by hand."""
    jd = 2461000.5
    prv_candidates = [
        detection(jd + 1, "g", 1000.0, 50.0),
        detection(jd + 3, "r", 2000.0, 60.0),  # the alert
        detection(jd + 9, "i", 3000.0, 70.0),  # later than the alert
    ]
    fp_hists = [
        forced(jd + 0, "u", 10.0, 40.0),
        forced(jd + 1, "g", 990.0, 55.0),  # duplicate of a detection
        forced(jd + 2, "z", None, 45.0),  # null flux
        forced(jd + 2, "y", -30.0, 80.0),
        forced(jd + 3, "r", 1990.0, 65.0),  # duplicate of the alert's detection
        forced(jd + 9, "i", 2990.0, 75.0),  # later than the alert
    ]
    return make_alert(42, "lsst-obj-42", prv_candidates, fp_hists, alert_jd=jd + 3, alert_band="r")


def test_merge_photometry():
    """Detections and forced photometry merge, deduplicated, without nulls or future points."""
    merged = merge_photometry(_small_alert())
    jd = 2461000.5

    np.testing.assert_array_equal(merged["jd"], [jd + 0, jd + 1, jd + 2, jd + 3])
    assert merged["band"].tolist() == ["u", "g", "y", "r"]
    np.testing.assert_array_equal(merged["flux"], [10.0, 1000.0, -30.0, 2000.0])
    np.testing.assert_array_equal(merged["flux_err"], [40.0, 50.0, 80.0, 60.0])
    assert merged["detected"].tolist() == [False, True, False, True]


def test_getters(consumer):
    """Getters convert to MJD and the ELAsTiCC FLUXCAL scale."""
    alert = decode_alert(encode_alert(_small_alert()))

    assert consumer.get_candid(alert) == 42
    assert consumer.get_objectId(alert) == "lsst-obj-42"
    np.testing.assert_array_equal(
        consumer.get_time(alert), np.array([0.0, 1.0, 2.0, 3.0]) + 2461000.5 - MJD_OFFSET
    )
    assert consumer.get_time(alert).dtype == np.float64
    np.testing.assert_allclose(
        consumer.get_flux(alert), np.array([10.0, 1000.0, -30.0, 2000.0]) * 0.027542, rtol=1e-4
    )
    np.testing.assert_allclose(
        consumer.get_flux_err(alert), np.array([40.0, 50.0, 80.0, 60.0]) * NJY_TO_FLUXCAL
    )
    assert consumer.get_band(alert).tolist() == ["u", "g", "y", "r"]
    assert consumer.get_detected(alert).tolist() == [False, True, False, True]


def test_realistic_alert(consumer):
    """A full alert: all six bands, sorted, nothing after the alert, every detection kept."""
    alert = decode_alert(encode_alert(make_light_curve_alert(np.random.default_rng(3), 7, "lsst-obj-7")))
    alert_mjd = alert["candidate"]["jd"] - MJD_OFFSET

    time = consumer.get_time(alert)
    assert set(consumer.get_band(alert)) == set(LSST_BANDS)
    assert np.all(np.diff(time) >= 0)
    assert time.max() == alert_mjd
    assert np.isfinite(consumer.get_flux(alert)).all()
    n_detections = sum(p["jd"] <= alert["candidate"]["jd"] for p in alert["prv_candidates"])
    assert consumer.get_detected(alert).sum() == n_detections


def test_empty_forced_history(consumer):
    """With no forced photometry, the light curve is the detections."""
    alert = decode_alert(
        encode_alert(make_light_curve_alert(np.random.default_rng(4), 8, "lsst-obj-8", empty_forced=True))
    )

    assert alert["fp_hists"] == []
    assert consumer.get_detected(alert).all()
    assert len(consumer.get_time(alert)) == sum(
        p["jd"] <= alert["candidate"]["jd"] for p in alert["prv_candidates"]
    )


def test_collate(consumer):
    """Per-point fields collate padded, with masks."""
    alerts = [decode_alert(encode_alert(alert)) for alert in _alerts()[:2]]
    samples = [{field: getattr(consumer, f"get_{field}")(alert) for field in FIELDS} for alert in alerts]

    for field in FIELDS:
        collated = getattr(consumer, f"collate_{field}")(samples)
        assert collated[f"{field}_mask"].sum(axis=1).tolist() == [len(s[field]) for s in samples]


class RecordingWriter(HyraxAlertsBaseWriter):
    """Collects every record process_alerts writes."""

    records: list[dict] = []

    def write(self, result_batch: list[dict]):
        """Keep the batch."""
        RecordingWriter.records.extend(result_batch)


@pytest.fixture
def recorded():
    """An empty record buffer, cleared afterwards."""
    RecordingWriter.records.clear()
    yield RecordingWriter.records
    RecordingWriter.records.clear()


def _write_config(tmp_path, weights, reconstruction_grid):
    config_file = tmp_path / "alerts_config.toml"
    config_file.write_text(
        f"""
[general]
results_dir = "{tmp_path / "results"}"

[data_loader]
batch_size = 2

[data_set.KafkaStreamDataset]
topics = "^babamul\\\\.lsst\\\\..*"
batch_flush_timeout = 0.0

[data_request.infer_stream.data]
dataset_class = "test_babamul_lsst.FakeBabamulLsstConsumer"
data_location = "kafka://localhost:9092"
primary_id_field = "candid"
fields = ["time", "flux", "flux_err", "band", "detected"]

[model]
name = "hyrax_seldon.models.seldon.HyraxSeldon"

[model.HyraxSeldon]
seldon_config = "{weights.parent.parent / "hparams.yaml"}"
reconstruction_grid = {str(reconstruction_grid).lower()}

[infer_stream]
model_weights_file = "{weights}"
save_model_output = false

[hyrax_alerts]
reject_output_root = false

[hyrax_alerts.writers.recorder]
writer_class = "RecordingWriter"
"""
    )
    return config_file


EXPECTED_KEYS = {
    "reconstructed",
    "error",
    "z_mean",
    "z_sigma",
    "z_var",
    "z_sample",
    "z",
    "decoder_inputs",
    "memory",
    "parameters_global_offset",
    "parameters_global_amplitude",
    "parameters_redshift",
    "class_idx",
    "class_emb",
    "elbo",
    "pad_mask_full",
    "short_lightcurve",
}


@pytest.mark.parametrize("reconstruction_grid", [False, True])
def test_process_alerts_end_to_end(tmp_path, seldon_checkpoint, recorded, reconstruction_grid):
    """process_alerts streams Babamul LSST alerts through HyraxSeldon to the writers.

    Every alert gets its own result with every output key; each result's point count matches
    its own alert's light curve, so a result paired with the wrong alert would be caught.
    """
    from hyrax_alerts.process_alerts import process_alerts

    process_alerts(config_filepath=str(_write_config(tmp_path, seldon_checkpoint, reconstruction_grid)))

    alerts = {alert["candid"]: alert for alert in _alerts()}
    assert sorted(int(record["object_id"]) for record in recorded) == sorted(alerts)

    expected = EXPECTED_KEYS | (
        {"grid_time", "grid_reconstructed", "grid_error"} if reconstruction_grid else set()
    )
    for record in recorded:
        result = record["__hyrax_result"]
        assert set(result) == expected

        n_points = len(merge_photometry(alerts[int(record["object_id"])])["jd"])
        assert int(np.asarray(result["pad_mask_full"]).sum()) == min(n_points, 512)
        assert np.asarray(result["reconstructed"]).shape == (512, 1)
        assert np.isfinite(np.asarray(result["z_mean"])).all()
        if reconstruction_grid:
            assert np.asarray(result["grid_reconstructed"]).shape == (6, 64)
