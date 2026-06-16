
import os
from TextSummarizer.custom_logging import logger
from TextSummarizer.entity import DataValidationConfig

import os
from pathlib import Path

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_required_files(self) -> bool:
        try:
            validation_status = True  # Assume valid first

            # Determine dataset directory from configuration. The DataIngestion pipeline
            # places the unzipped samsum dataset under data_ingestion/unzipped_data/samsum_dataset
            # Use the configured STATUS_FILE path's parent to compute consistent artifact roots
            # If the config provides absolute paths, respect them.
            from pathlib import Path

            # If the config includes a root or explicit path to the unzipped data, prefer it.
            # Look for an environment where DataIngestion wrote to an artifacts path defined in YAML.
            # Fallback to relative path for backward compatibility.
            data_dir = None
            # Try common config attributes
            try:
                # If config contains a STATUS_FILE path, derive artifacts root
                status_file = Path(self.config.STATUS_FILE)
                artifacts_root = status_file.parent.parent  # status_file is data_validation/status.txt
                candidate = artifacts_root / 'data_ingestion' / 'unzipped_data' / 'samsum_dataset'
                if candidate.exists():
                    data_dir = candidate
            except Exception:
                data_dir = None

            # If not found yet, try to use environment/config default relative path
            if data_dir is None:
                candidate = Path("artifacts") / "data_ingestion" / "unzipped_data" / "samsum_dataset"
                data_dir = candidate

            all_files = [p.name for p in Path(data_dir).iterdir()] if Path(data_dir).exists() else []

            for file in self.config.ALL_REQUIRED_FILES:  # Check REQUIRED files only
                if file not in all_files:
                    validation_status = False
                    break   # One missing file = fail immediately

            # Ensure status directory exists before writing
            status_path = Path(self.config.STATUS_FILE)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            # Write the status ONCE (not inside loop)
            with status_path.open("w") as s:
                s.write(f"validation_status: {validation_status}\n")

            return validation_status

        except Exception as e:
            raise e
