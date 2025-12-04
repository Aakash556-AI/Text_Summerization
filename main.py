from TextSummarizer.custom_logging import logger
from TextSummarizer.config.configuration import ConfigurationManager
from TextSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

stage_name = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {stage_name} started <<<<<<")

    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()

    pipeline = DataIngestionTrainingPipeline(config=data_ingestion_config)
    pipeline.main()

    logger.info(f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e
