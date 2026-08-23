"""Repository-wide integration contracts for registered distortion metrics."""

from __future__ import annotations

import inspect
import re

import numpy as np
import pytest

from zadu import MEASURE, ZADU, make_spec
from zadu import measures as measure_package
from zadu.registry import (
    METRIC_BY_ALIAS,
    METRIC_BY_ID,
    METRIC_LOOKUP,
    METRICS,
    MetricDefinition,
)

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_RESOURCE_BACKED_METRICS = tuple(
    definition for definition in METRICS if definition.resources
)


def _contract_params(definition: MetricDefinition) -> dict:
    """Return quick, deterministic parameters using only public options."""

    params = {}
    if "k" in definition.user_params:
        params["k"] = 5
    if "iteration" in definition.user_params:
        params["iteration"] = 2
    if "clustering_strategy" in definition.user_params:
        params["clustering_strategy"] = "kmeans"
    if "random_state" in definition.user_params:
        params["random_state"] = 0
    if "n_jobs" in definition.user_params:
        params["n_jobs"] = 1
    if "n_triplets" in definition.user_params:
        params["n_triplets"] = 20
    if "random_seed" in definition.user_params:
        params["random_seed"] = 0
    return params


def _sample_data():
    rng = np.random.default_rng(2026)
    orig = rng.normal(size=(48, 8))
    emb = orig[:, :2] + 0.05 * rng.normal(size=(48, 2))
    labels = np.repeat(np.arange(4), 12)
    return orig, emb, labels


def _assert_score_dicts_close(actual, expected):
    assert isinstance(actual, dict)
    assert isinstance(expected, dict)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert isinstance(actual[key], (int, float))
        assert np.isfinite(actual[key])
        assert actual[key] == pytest.approx(expected[key], rel=1e-10, abs=1e-12)


def test_metric_registry_has_one_complete_public_contract_per_metric():
    ids = [definition.id for definition in METRICS]
    aliases = [definition.alias for definition in METRICS]

    assert len(ids) == len(set(ids))
    assert len(aliases) == len(set(aliases))
    assert set(METRIC_BY_ID) == set(ids)
    assert set(METRIC_BY_ALIAS) == set(aliases)
    assert {member.value for member in MEASURE} == set(ids)

    for definition in METRICS:
        assert _IDENTIFIER.fullmatch(definition.id)
        assert _IDENTIFIER.fullmatch(definition.alias)
        assert METRIC_BY_ID[definition.id] is definition
        assert METRIC_BY_ALIAS[definition.alias] is definition
        assert METRIC_LOOKUP[definition.id] is definition
        assert METRIC_LOOKUP[definition.alias] is definition
        assert ZADU.ABBREVIATIONS[definition.alias] == definition.id
        assert definition.id in measure_package.__all__

        module = definition.load()
        measure = getattr(module, "measure", None)
        assert callable(measure), f"{definition.id} must expose measure()"
        assert inspect.getdoc(measure), f"{definition.id}.measure needs a docstring"

        signature = inspect.signature(measure)
        parameters = signature.parameters
        expected = set(definition.inputs) | set(definition.user_params)
        expected.update(requirement.argument for requirement in definition.resources)
        if definition.needs_label:
            expected.add("label")
        if definition.supports_local:
            expected.add("return_local")
        assert expected <= set(parameters), (
            f"{definition.id} registry fields are missing from measure(): "
            f"{sorted(expected - set(parameters))}"
        )
        assert ("label" in parameters) is definition.needs_label
        assert ("return_local" in parameters) is definition.supports_local
        assert all(
            parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            for parameter in parameters.values()
        ), f"{definition.id}.measure must use an explicit signature"

        for name in definition.user_params:
            assert parameters[name].default is not inspect.Parameter.empty, (
                f"Public parameter {definition.id}.{name} needs a default so a "
                "minimal metric specification remains valid"
            )

        typed_spec = make_spec(MEASURE(definition.id), **_contract_params(definition))
        assert typed_spec["id"] == definition.id


@pytest.mark.parametrize("definition", METRICS, ids=lambda metric: metric.alias)
def test_registered_metric_direct_and_scheduled_interfaces_match(definition):
    orig, emb, labels = _sample_data()
    params = _contract_params(definition)
    direct_args = {}
    if "orig" in definition.inputs:
        direct_args["orig"] = orig
    if "emb" in definition.inputs:
        direct_args["emb"] = emb
    if definition.needs_label:
        direct_args["label"] = labels

    direct = definition.load().measure(**direct_args, **params)
    runner = ZADU(
        [{"id": definition.alias, "params": params}],
        orig,
    )
    scheduled = runner.measure(emb, labels)[0]

    _assert_score_dicts_close(scheduled, direct)


@pytest.mark.parametrize(
    "definition",
    _RESOURCE_BACKED_METRICS,
    ids=lambda metric: metric.alias,
)
def test_registered_metric_declared_resources_are_shared_by_the_dag(definition):
    orig, emb, labels = _sample_data()
    params = _contract_params(definition)
    repeated_spec = {"id": definition.alias, "params": params}
    runner = ZADU([repeated_spec, repeated_spec], orig)

    runner.measure(emb, labels)

    resources = runner.last_run_info["resources"]
    assert resources, f"{definition.id} declares resources but planned none"
    for resource in resources:
        assert resource["consumer_count"] == 2
        assert resource["consumers"] == [definition.id, definition.id]
