from __future__ import annotations

import pytest

from resilient_traffic.config import load_config


@pytest.fixture
def config():
    return load_config("quick")

