
from pathlib import Path

# Absolute path for config.yaml is kept as-is (can be updated if repo moved).
CONFIG_FILE_PATH: Path = Path("/Users/akashkumarsinha/Desktop/Text_summerization/Text_Summerization/config/config.yaml")

# Make PARAMS_FILE_PATH robust by resolving it relative to the package root
# This ensures callers can run the script from the workspace root or any cwd
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PARAMS_FILE_PATH: Path = PACKAGE_ROOT / "params.yaml"
