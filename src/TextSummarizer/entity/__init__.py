
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_download_url: str
    raw_data_dir: Path
    local_data_file: Path
    unzipped_data_dir: Path
    