import argparse

import wandb

from config import get_cfg_defaults
from models.googlenetmodel import GoogLeNetModel
from utils.model_utils import setup_seed


parser = argparse.ArgumentParser()

# parser.add_argument("--model", "-m", default="FNN")
# parser.add_argument("--learning_rate", "-l", default=0.001, type=float)
# parser.add_argument("--hidden", "-hd", default=256, type=int)
# parser.add_argument("--batchsize", "-b", default=512, type=int)
parser.add_argument("--epoch", "-e", default=20, type=int)
parser.add_argument("--wandb", "-w", default=False, action="store_true")


args = parser.parse_args()

if __name__ == "__main__":
    setup_seed(22)
    # wandb.init()
    cfg = get_cfg_defaults()
    # cfg.MODEL.NAME = args.model
    cfg.WANDB.OPEN = args.wandb
    # cfg.TRAIN.LEARNING_RATE = args.learning_rate
    # cfg.TRAIN.HIDDEN_SIZE = args.hidden
    # cfg.TRAIN.BATCH_SIZE = args.batchsize
    cfg.TRAIN.NUM_EPOCH = args.epoch
    print(cfg)
    print()

    assert cfg.MODEL.NAME.lower() in ["googlenet", "softmax"]

    trainer = GoogLeNetModel(cfg)
    # trainer.run()
    trainer.sweep()



    pass