"""Test high‑level training utilities."""

import os
import torch
import pytest

from conftest import ignore_sensitivity_warning, ignore_pin_memory_mps_warning

from hippynn.experiment.routines import (
    train_model,
    setup_and_profile,
    setup_and_train,
)


@ignore_sensitivity_warning
@ignore_pin_memory_mps_warning
def test_train_model(example_training_setup, temporary_directory):
    training_modules, database, setup_params = example_training_setup
    from hippynn.experiment.routines import setup_training

    training_modules, controller, metric_tracker = setup_training(training_modules=training_modules, setup_params=setup_params)
    result_tracker = train_model(
        training_modules=training_modules,
        database=database,
        controller=controller,
        metric_tracker=metric_tracker,
        callbacks=None,
        batch_callbacks=None,
        store_all_better=False,
        store_best=False,
        store_every=0,
        quiet=True,
    )
    assert result_tracker is not None


@ignore_sensitivity_warning
@ignore_pin_memory_mps_warning
def test_setup_and_profile(example_training_setup, temporary_directory):
    training_modules, database, setup_params = example_training_setup
    trace_file = setup_and_profile(
        training_modules=training_modules, database=database, setup_params=setup_params, profile_epochs=1, batches_per_epoch=1
    )
    assert os.path.isfile(trace_file)


@ignore_sensitivity_warning
@ignore_pin_memory_mps_warning
def test_setup_and_train(example_training_setup, temporary_directory):
    training_modules, database, setup_params = example_training_setup
    metric_tracker = setup_and_train(training_modules=training_modules, database=database, setup_params=setup_params)
    assert metric_tracker is not None


@ignore_sensitivity_warning
@ignore_pin_memory_mps_warning
def test_test_model(example_training_setup, temporary_directory):
    # this function needs to be imported in an inner scope to avoid pytest automatically running it.
    from hippynn.experiment.routines import test_model

    training_modules, database, setup_params = example_training_setup
    evaluator = training_modules.evaluator
    # The evaluator defaults to device_fallback() (e.g. cuda), but the untrained model is still on CPU.
    evaluator.model_device = next(evaluator.model.parameters()).device
    metric_tracker = test_model(database, evaluator, when="example", batch_size=20)
    assert metric_tracker is not None
