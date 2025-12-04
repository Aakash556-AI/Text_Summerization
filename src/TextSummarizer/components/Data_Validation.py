
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

            data_dir = os.path.join("artifacts", "data_ingestion", "unzipped_data","samsum_dataset")
            all_files = os.listdir(data_dir)

            for file in self.config.ALL_REQUIRED_FILES:  # Check REQUIRED files only
                if file not in all_files:
                    validation_status = False
                    break   # One missing file = fail immediately

            # Ensure status directory exists before writing
            status_dir = os.path.dirname(self.config.STATUS_FILE)
            os.makedirs(status_dir, exist_ok=True)

            # Write the status ONCE (not inside loop)
            with open(self.config.STATUS_FILE, "w") as s:
                s.write(f"validation_status: {validation_status}\n")

            return validation_status

        except Exception as e:
            raise e
