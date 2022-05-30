from mindspore import load_checkpoint
from nets.googlenet import GoogLeNet_backbone
from nets.inceptionv3 import InceptionV3_backbone
from utils.model_utils import extract_features
from utils.dataset_utils import create_dataset






if __name__ == "__main__":
    batch_size = 128
    # mode = "train"
    mode = "test"
    eval_set = create_dataset(f"./dataset/{mode}.list", 299, train=False, batch_size=batch_size, shuffle=False)
    # googlenet_backbone = GoogLeNet_backbone()
    inceptionv3_backbone = InceptionV3_backbone()
    load_checkpoint("./checkpoints/model_InceptionV3_best_param_backbone.ckpt", inceptionv3_backbone)
    extract_features(inceptionv3_backbone, eval_set, "InceptionV3", f"./checkpoints/features/{mode}", batch_size)


