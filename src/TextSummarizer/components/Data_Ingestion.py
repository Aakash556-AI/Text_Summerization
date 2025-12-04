
import os
import urllib.request as request
import zipfile
from TextSummarizer.custom_logging import logger
from TextSummarizer.utils.common import get_size
from TextSummarizer.entity import DataIngestionConfig

class dataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> str:
        if not os.path.exists(self.config.local_data_file):
            logger.info("Downloading file...")
            filename, headers = request.urlretrieve(
                self.config.dataset_download_url, self.config.local_data_file
            )
            logger.info(
                f"File downloaded successfully! File location: {filename}, size: {get_size(self.config.local_data_file)}"
            )
        else:
            logger.info("File already exists. Skipping download.")

       
    
    def extract_zip_file(self, zip_file_path: str):
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(self.config.unzipped_data_dir)
        logger.info("Extraction completed.")
       
