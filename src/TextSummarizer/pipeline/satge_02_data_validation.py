
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.Data_Validation import DataValidation
from TextSummarizer.custom_logging import logger


class DataValidationTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
        data_validation = DataValidation(config=self.config)
        data_validation.validate_required_files()
