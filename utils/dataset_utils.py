import os
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.common import dtype as mstype
import mindspore.dataset as ds
import multiprocessing

from utils.model_utils import setup_seed

setup_seed(22)

class PetData:
    def __init__(self, file_list):
        with open(file_list) as f:
            self.img_list = f.read().splitlines()
        f.close()

    def __getitem__(self, item):
        img_path = self.img_list[item]
        img = Image.open(img_path)
        label = 0 if img_path.split("/")[-1].split(".")[0] == "cat" else 1

        return img, label

    def __len__(self):
        return len(self.img_list)


def gen_dataset(data_path, save_path, test_size=0.2, random_seed=22):
    """
    ! arc decompress dogs-vs-cats.zip
    :param data_path: /root/autodl-tmp/PycharmProjects/code/dataset/train
    :param save_path: ../dataset
    :param test_size:
    :param random_seed:
    :return:
    """

    cat_list = glob.glob(os.path.join(data_path, "cat*"))
    dog_list = glob.glob(os.path.join(data_path, "dog*"))
    X = np.append(cat_list, dog_list)
    y = np.append(np.ones(len(cat_list)), np.zeros(len(dog_list)))

    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)

    with open(os.path.join(save_path, "train.list"), "w") as f:
        f.write("\n".join(X_train))
    f.close()

    with open(os.path.join(save_path, "test.list"), "w") as f:
        f.write("\n".join(X_test))
    f.close()

    print("train file in ", os.path.join(save_path, "train.list"))
    print("test file in ", os.path.join(save_path, "test.list"))


def create_dataset(file_list, size, train=True, batch_size=1, shuffle=True):
    dataset_generator = PetData(file_list)

    if shuffle:
        cores = max(min(multiprocessing.cpu_count(), 8), 1)
        dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True, num_parallel_workers=cores)
    else:
        dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=False, num_parallel_workers=1)

    RGB_mean = [124.479, 116.011, 106.281]
    RGB_std = [66.734, 65.031, 65.683]

    if train:
        trans = [
            CV.Resize([320, 320]),
            CV.RandomCrop([size, size]),
            CV.Normalize(RGB_mean, RGB_std),
            CV.RandomHorizontalFlip(),
            CV.HWC2CHW()
        ]
    else:
        trans = [
            CV.Resize([size, size]),
            CV.Normalize(RGB_mean, RGB_std),
            CV.HWC2CHW()
        ]

    typecast_op = C.TypeCast(mstype.int32)

    dataset = dataset.map(input_columns='label', operations=typecast_op)
    dataset = dataset.map(input_columns='image', operations=trans)

    dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset


def create_dataset_aug(file_list, size, train=True, batch_size=1, shuffle=True):
    dataset_generator = PetData(file_list)

    if shuffle:
        cores = max(min(multiprocessing.cpu_count(), 8), 1)
        dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=True, num_parallel_workers=cores)
    else:
        dataset = ds.GeneratorDataset(dataset_generator, ["image", "label"], shuffle=False, num_parallel_workers=1)

    RGB_mean = [124.479, 116.011, 106.281]
    RGB_std = [66.734, 65.031, 65.683]

    if train:
        trans = [
            CV.Resize([320, 320]),
            CV.RandomCrop([size, size]),
            CV.RandomColorAdjust(0.4, 0.4, 0.4, 0.1),
            CV.RandomRotation(degrees=5),
            CV.Normalize(RGB_mean, RGB_std),
            CV.RandomHorizontalFlip(),
            CV.RandomVerticalFlip(),
            CV.HWC2CHW()
        ]
    else:
        trans = [
            CV.Resize([size, size]),
            CV.Normalize(RGB_mean, RGB_std),
            CV.HWC2CHW()
        ]

    typecast_op = C.TypeCast(mstype.int32)

    dataset = dataset.map(input_columns='label', operations=typecast_op)
    dataset = dataset.map(input_columns='image', operations=trans)

    dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset


if __name__ == '__main__':
    gen_dataset("/root/autodl-tmp/PycharmProjects/code/dataset/train", "../dataset")
