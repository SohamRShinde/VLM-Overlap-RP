"""Fails loudly if the package layout or pytest pythonpath config breaks."""

import circuit_overlap
from circuit_overlap import metrics


def test_package_imports():
    assert circuit_overlap.__doc__
    assert hasattr(metrics, "jaccard")
