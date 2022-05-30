from yacs.config import CfgNode as CN

_C = CN()

_C.WANDB = CN()
_C.WANDB.OPEN = True
_C.WANDB.PROJECT_NAME = "DLFIN"
_C.WANDB.ENTITY = "cugcuiyc"
_C.WANDB.RESUME = False
_C.WANDB.LOG_DIR = ""
_C.WANDB.SWEEP_CONFIG = "./config.json"

_C.MODEL = CN()
_C.MODEL.NAME = "SVM"

_C.DATASET = CN()
_C.DATASET.NAME = "dogs.vs.cats"

_C.TRAIN = CN()
_C.TRAIN.TRAIN_LIST = "./dataset/train.list"
_C.TRAIN.TEST_LIST = "./dataset/test.list"
_C.TRAIN.SAVE_PATH = "./checkpoints"
_C.TRAIN.FEATURE_PATH = "./checkpoints/features"


def get_cfg_defaults():
    return _C.clone()


cfg = _C