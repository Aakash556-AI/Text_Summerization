
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_download_url: str
    raw_data_dir: Path
    local_data_file: Path
    unzipped_data_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: Path
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    transformed_data_dir: Path
    tokenizer_name: str
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt_dir: Path
    num_train_epochs: float
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    save_steps: float
    gradient_accumulation_steps: int
    report_to: str

@dataclass(frozen=True)
class ModelEvoluationConfig:
  root_dir: Path
  data_path: Path
  model_path: Path
  tokenizer_path: Path
  matric_file_name: Path