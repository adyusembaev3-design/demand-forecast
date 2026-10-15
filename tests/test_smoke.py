"""Step 1 smoke test: the project imports and its folders exist."""
from src import config


def test_paths_exist():
    assert config.DATA_RAW.exists()
    assert config.DATA_PROCESSED.exists()


def test_get_env_default():
    assert config.get_env("DOES_NOT_EXIST", required=False, default="x") == "x"
