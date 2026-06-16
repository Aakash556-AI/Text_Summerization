import os
from pathlib import Path
from TextSummarizer.utils.common import read_yaml


def test_read_params_yaml_from_workspace_root():
    # repo_root is two levels up from this file when tests live under Text_Summerization/tests
    repo_root = Path(__file__).resolve().parents[2]
    params_path = repo_root / "Text_Summerization" / "params.yaml"
    cfg = read_yaml(params_path)
    assert cfg is not None
    # basic smoke check: params should contain TrainingArguments
    assert "TrainingArguments" in cfg
