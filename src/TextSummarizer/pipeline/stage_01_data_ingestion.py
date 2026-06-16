from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.Data_Ingestion import dataIngestion
from TextSummarizer.custom_logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
        data_ingestion = dataIngestion(config=self.config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file(
            zip_file_path=self.config.local_data_file
        )
