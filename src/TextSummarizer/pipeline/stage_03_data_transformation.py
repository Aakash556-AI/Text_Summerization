
from TextSummarizer.components.Data_Transformation import DataTransformation
from TextSummarizer.custom_logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
        data_transformation = DataTransformation(self.config)
        data_transformation.convert()
        logger.info("Data Transformation Completed")
from TextSummarizer.components.Data_Transformation import DataTransformation
from TextSummarizer.custom_logging import logger


class DataTransformationTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
         data_transformation = DataTransformation(self.config)
         data_transformation.convert()
         logger.info("Data Transformation Completed")
