
from enum import Enum


class RUN_MODE(Enum):
    TRAIN = "train"
    TRAIN_CONTINUOUSLY = "train_continuously"
    EVALUATE_ONLY = "evaluate_only"


class CommonVars:
    def __init__(self) -> None:
        self.outputs_dir = "outputs"
        self.logs_dir = "outputs/logs"
        self.stats_dir = "outputs/stats"
        self.data_dir = "data"
        self.models_dir = "outputs/models"
        self.configs_dir = "configs"
        self.model_name = ""


common_vars = CommonVars()
