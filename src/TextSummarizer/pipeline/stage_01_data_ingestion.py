from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.Data_Ingestion import dataIngestion
from TextSummarizer.custom_logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
        config_manager = ConfigurationManager()

        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = dataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file(
            zip_file_path=data_ingestion_config.local_data_file
        )

    