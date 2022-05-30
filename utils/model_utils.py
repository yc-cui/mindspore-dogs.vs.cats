import numpy as np
import itertools
import matplotlib.pyplot as plt
import mindspore
import random
import os
import shutil
from mindspore import Model, load_checkpoint
from mindspore.common import set_seed

from nets.googlenet import GoogLeNet_backbone


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    mindspore.set_seed(seed)
    set_seed(seed)



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    https://github.com/gcamfer/Anomaly-ReactionRL/blob/master/Notebooks/AE_RL_NSL_KDD.ipynb
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, 2)
    else:
        pass
    plt.cla()
    plt.close("all")
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.0f' if cm[i, j] == 0 else fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt

# save_path: ./checkpoints/features/train
# model_name: GoogLeNet
def extract_features(net, dataset, model_name, save_path, batch_size):

    features_folder = os.path.join(save_path, model_name + "_" + str(batch_size)+ "_features")
    if os.path.exists(features_folder):
        shutil.rmtree(features_folder)
    os.makedirs(features_folder)

    step_size = dataset.get_dataset_size()

    model = Model(net)

    for i, data in enumerate(dataset.create_dict_iterator()):

        image = data['image']
        label = data['label']

        features = model.predict(image)
        # features = image

        features_path = os.path.join(features_folder, f'feature_{i}.npy')
        label_path = os.path.join(features_folder, f'label_{i}.npy')
        np.save(features_path, features.asnumpy())
        np.save(label_path, label.asnumpy())

        print(f"Complete the batch {i + 1}/{step_size}")

    return step_size


