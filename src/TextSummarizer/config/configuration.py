from TextSummarizer.constants import *
from TextSummarizer.utils.common import read_yaml, create_directories

from TextSummarizer.entity import (DataIngestionConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_download_url=config.dataset_download_url,
            raw_data_dir=Path(config.raw_data_dir),
            local_data_file=Path(config.local_data_file),
            unzipped_data_dir=Path(config.unzipped_data_dir),
        )

        return data_ingestion_config