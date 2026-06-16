
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.components.model_trainer import ModelTrainer
from TextSummarizer.custom_logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self, config):
        self.config = config

    def main(self):
         model_trainer = ModelTrainer(self.config)
         model_trainer.train()
         logger.info("Model Trainer Training Completed")
