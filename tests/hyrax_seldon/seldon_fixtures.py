"""Synthetic ELAsTiCC data and SELDON run directories for tests.

:func:`make_elasticc_parquet` writes light curves in the ELAsTiCC parquet schema SELDON's
``FlexibleLightCurveDataLoader`` reads. :func:`make_seldon_run_dir` produces a SELDON run
directory the way SELDON's ``train.py`` does - the target yaml, SELDON's own data module
computing the dataset statistics, a randomly initialized ``SELDONEnergyClassifierExperiment3``,
a TensorBoard ``hparams.yaml`` and a Lightning ``ModelCheckpoint`` ``.ckpt`` - so that the
model is tested against SELDON's real file formats.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_CONFIG = "SELDON_2.0_S_E_UQ_C_with_transmission_delta_refactored.yaml"
CLUSTER_DATA_DIR = "/projects/ncsa/caps/skai/software/Transient-Foundation-Model/TAE/data/"

# One raw ELAsTiCC class for each of the ten classes in the target config's class_filter.
ELASTICC_CLASSES = [
    "SNIa-SALT3",
    "SNII-NMF",
    "SNIIb+HostXT_V19",
    "SNIIn-MOSFIT",
    "SNIb-Templates",
    "SNIax",
    "SLSN-I+host",
    "KN_K17",
    "ILOT",
    "PISN",
]

ELASTICC_BANDS = ["u", "g", "r", "i", "z", "Y"]
LSST_BANDS = ["u", "g", "r", "i", "z", "y"]

DETECTION_FLAG = 4096
FIRST_DETECTION_FLAG = 2048
SATURATION_FLAG = 1024


def make_light_curve(rng, n_points, mjd0=60000.0, n_saturated=0):
    """Return one synthetic ELAsTiCC light curve as a dict of per-point arrays.

    Times are strictly increasing (no ties), so every sort order agrees. Points with
    ``FLUXCAL > 5 * FLUXCALERR`` are detections; ``n_saturated`` detections are also
    flagged saturated.
    """
    mjd = mjd0 + np.cumsum(rng.uniform(0.3, 3.0, n_points))
    peak = mjd[0] + rng.uniform(0.3, 0.7) * (mjd[-1] - mjd[0])
    amplitude = rng.uniform(500.0, 5000.0)
    width = rng.uniform(10.0, 40.0)
    flux_err = rng.uniform(5.0, 30.0, n_points)
    flux = amplitude * np.exp(-0.5 * ((mjd - peak) / width) ** 2) + rng.normal(0.0, 1.0, n_points) * flux_err

    detected = flux > 5 * flux_err
    photflag = np.where(detected, DETECTION_FLAG, 0)
    if detected.any():
        photflag[np.argmax(detected)] |= FIRST_DETECTION_FLAG
    if n_saturated:
        saturated = rng.choice(np.where(detected)[0], n_saturated, replace=False)
        photflag[saturated] |= SATURATION_FLAG

    return {
        "MJD": mjd,
        "FLUXCAL": flux.astype(np.float32),
        "FLUXCALERR": flux_err.astype(np.float32),
        "BAND": rng.choice(ELASTICC_BANDS, n_points).astype(object),
        "PHOTFLAG": photflag.astype(np.int64),
    }


def make_elasticc_parquet(path, n_objects=20, min_length=90, max_length=160, seed=0, n_saturated=2):
    """Write ``n_objects`` synthetic light curves in the ELAsTiCC parquet schema.

    Every object is bright enough to have at least SELDON's ``min_points`` (32) detections,
    and the classes cycle through :data:`ELASTICC_CLASSES` so all ten training classes are
    present.

    Returns
    -------
    pandas.DataFrame
        The table that was written.
    """
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n_objects:
        index = len(rows)
        light_curve = make_light_curve(
            rng, int(rng.integers(min_length, max_length + 1)), n_saturated=n_saturated
        )
        detections = (light_curve["PHOTFLAG"] & DETECTION_FLAG) & ~(light_curve["PHOTFLAG"] & SATURATION_FLAG)
        if np.count_nonzero(detections) < 40:
            continue
        rows.append(
            {
                "SNID": np.int64(1000 + index),
                "ELASTICC_class": ELASTICC_CLASSES[index % len(ELASTICC_CLASSES)],
                "REDSHIFT_FINAL": float(rng.uniform(0.05, 1.0)),
                "REDSHIFT_FINAL_ERR": 0.01,
                **light_curve,
            }
        )
    table = pd.DataFrame(rows)
    table.to_parquet(path)
    return table


def random_light_curve(rng, n_points, detected_fraction=0.9):
    """Return raw per-point arrays: float32 MJD (as SELDON stores it), flux, error, band, PHOTFLAG."""
    mjd = (60000.0 + np.cumsum(rng.uniform(0.3, 3.0, n_points))).astype(np.float32)
    flux = rng.normal(800.0, 900.0, n_points).astype(np.float32)
    flux_err = rng.uniform(5.0, 40.0, n_points).astype(np.float32)
    band = rng.choice(LSST_BANDS, n_points)
    photflag = np.where(rng.random(n_points) < detected_fraction, DETECTION_FLAG, 0).astype(np.int64)
    return mjd, flux, flux_err, band, photflag


def collate_light_curves(light_curves):
    """Collate raw light curves the way hyrax_seldon's datasets do: padded, with masks."""
    length = max(len(lc[0]) for lc in light_curves)
    batch = {}
    for name, index, pad in [
        ("time", 0, 0.0),
        ("flux", 1, 0.0),
        ("flux_err", 2, 0.0),
        ("band", 3, ""),
        ("photflag", 4, 0),
    ]:
        values = [lc[index] for lc in light_curves]
        padded = np.full((len(values), length), pad, dtype=np.asarray(values[0]).dtype)
        mask = np.zeros((len(values), length), dtype=bool)
        for row, value in enumerate(values):
            padded[row, : len(value)] = value
            mask[row, : len(value)] = True
        batch[name] = padded
        batch[f"{name}_mask"] = mask
    return batch


def prepare_inputs(batch, config):
    """Run ``HyraxSeldon.prepare_inputs`` on a batch's ``"data"`` group, as a verb run does.

    ``config`` is installed in Hyrax's run context, where ``prepare_inputs`` reads it.
    """
    from hyrax.context import run_context

    from hyrax_seldon.models.seldon import HyraxSeldon

    with run_context("infer", config=config):
        return HyraxSeldon.prepare_inputs({"data": batch})


def target_yaml_text():
    """Return the target SELDON yaml with transmission paths pointing at the installed files."""
    import TAE

    tae_dir = Path(TAE.__file__).parent
    text = (tae_dir / "configs" / TARGET_CONFIG).read_text()
    return text.replace(CLUSTER_DATA_DIR, str(tae_dir / "data") + "/")


def make_seldon_run_dir(root, parquet_path, batch_size=2, seed=42):
    """Produce a SELDON run directory the way SELDON's ``train.py`` does.

    This follows ``train.py`` step for step, with a randomly initialized model: hydra's
    config loading is replaced by ``OmegaConf.load`` of the target yaml, the constructor
    ``compile()`` calls are undone (inductor does not run here), and the Lightning
    trainer runs one validation pass so that ``ModelCheckpoint`` writes the checkpoint
    exactly as in a real run.

    Returns
    -------
    Path
        The run directory, containing ``hparams.yaml`` and ``checkpoints/<name>.ckpt``.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import lightning.pytorch as pl
    import TAE.util  # noqa: F401  (registers OmegaConf resolvers)
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from omegaconf import OmegaConf, open_dict
    from TAE.datasets.dataset_factory import get_module as get_dataset
    from TAE.experiments.factory import get_module as get_experiment
    from TAE.models.factory import get_module
    from TAE.util import fill_config

    cfg = OmegaConf.create(target_yaml_text())
    cfg.dataset.config.lightcurve_path = [str(parquet_path)]
    cfg.dataset.config.batch_size = batch_size
    cfg.dataset.config.num_workers = 0
    cfg.dataset.config.pin_memory = False
    cfg.logging_params.save_dir = str(root)
    cfg.trainer_params.accelerator = "cpu"
    cfg.trainer_params.devices = 1
    cfg.trainer_params.strategy = "auto"
    cfg.trainer_params.max_epochs = 1
    cfg.logging_params.manual_seed = seed

    cfg = fill_config(cfg)
    config = OmegaConf.to_container(cfg)

    pl.seed_everything(seed)

    tb_logger = TensorBoardLogger(
        save_dir=cfg["logging_params"]["save_dir"], name=cfg["logging_params"]["name"]
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        filename="{epoch}-{val_loss:.2f}",
        verbose=False,
        save_last=False,
        every_n_epochs=1,
    )

    model = get_module(config["model_params"]["name"], config["model_params"]["config"])
    for module in model.modules():
        module._compiled_call_impl = None

    data = get_dataset(config["dataset"]["name"], config["dataset"]["config"])
    try:
        data.setup()

        config["dataset"]["config"]["flux_stats"] = data.flux_stats
        config["dataset"]["config"]["band_weights"] = data.band_weights
        config["dataset"]["config"]["class_map"] = data.class_map
        with open_dict(cfg):
            cfg.dataset.config.flux_stats = {
                key: float(value) if isinstance(value, np.float32) else value
                for key, value in data.flux_stats.items()
            }
            cfg.dataset.config.band_weights = {
                key: float(value) if isinstance(value, np.float64) else value
                for key, value in data.band_weights.items()
            }
            cfg.dataset.config.class_map = {
                key: int(value) if isinstance(value, np.float32) else value
                for key, value in data.class_map.items()
            }

        experiment = get_experiment(
            cfg["exp_params"]["name"],
            {"model": model, "data": data, **config["exp_params"]["config"]},
        )

        with open_dict(cfg["trainer_params"]):
            cfg["trainer_params"].pop("resume_from_checkpoint", None)

        tb_logger.log_hyperparams(params=cfg)

        runner = Trainer(
            enable_checkpointing=True,
            callbacks=[checkpoint_callback],
            logger=tb_logger,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            **cfg["trainer_params"],
        )
        runner.fit(experiment, data)
    finally:
        data.manager.shutdown()

    return Path(tb_logger.log_dir)
