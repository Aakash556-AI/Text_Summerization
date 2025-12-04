
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.Data_Validation import DataValidation
from TextSummarizer.custom_logging import logger


class DataValidationTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
        config_manager = ConfigurationManager()

        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        status = data_validation.validate_required_files()
        

    