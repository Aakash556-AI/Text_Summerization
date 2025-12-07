
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.Data_Transformation import DataTransformation
from TextSummarizer.custom_logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
         config = ConfigurationManager()
         data_transformation_config = config.get_data_transformation_config()
         data_transformation = DataTransformation(data_transformation_config)
         data_transformation.convert()
         logger.info("Data Transformation Completed")

