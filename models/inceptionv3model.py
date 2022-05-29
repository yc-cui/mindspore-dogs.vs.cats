import random

import wandb
import json
import numpy as np
from mindspore import nn, load_checkpoint
from sklearn.metrics import accuracy_score
from mindspore import context, save_checkpoint
from utils.model_utils import setup_seed
from tqdm import tqdm
from nets.inceptionv3 import InceptionV3
from utils.dataset_utils import create_dataset
import os

# os.environ["WANDB_CONSOLE"] = "off"
setup_seed(22)
device_target = context.get_context('device_target')
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)


# context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)


class InceptionV3Model:
    def __init__(self, opt):
        self.opt = opt
        self.model_name = "model_{}".format(self.opt.MODEL.NAME)

        batch_list = [8, 16, 32]
        self.train_set_dict = {i: create_dataset(self.opt.TRAIN.TRAIN_LIST, 299, train=True, batch_size=i, shuffle=False) for i in batch_list}
        self.train_set_iter_dict = {k: v.create_dict_iterator() for k, v in self.train_set_dict.items()}

        self.eval_train_set = create_dataset(self.opt.TRAIN.TRAIN_LIST, 299, train=False, batch_size=64, shuffle=False)
        self.eval_test_set = create_dataset(self.opt.TRAIN.TEST_LIST, 299, train=False, batch_size=64, shuffle=False)
        self.eval_train_set_iter = self.eval_train_set.create_dict_iterator()
        self.eval_test_iter = self.eval_test_set.create_dict_iterator()

        self.net = InceptionV3(2)
        self.global_max_acc = 0



    def run_sweep(self):
        setup_seed(22)
        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        with wandb.init(name=self.model_name,
                        config=self.opt,
                        notes=self.opt.WANDB.LOG_DIR,
                        resume=self.opt.WANDB.RESUME,
                        ) as run:
            config = wandb.config
            wandb.run.name = "_".join([self.model_name, config["optimizer"], str(config["lr"]), str(config["batch_size"])])

            num_epoch = self.opt.TRAIN.NUM_EPOCH

            init_weights_path = './checkpoints/init_{}.ckpt'.format(self.model_name)
            if os.path.exists(init_weights_path):
                print("loading existed initial weights [  {}  ] to net...".format(init_weights_path))
                load_checkpoint(init_weights_path, net=self.net)
            else:
                print("saving initial weights [  {}  ] from net...".format(init_weights_path))
                save_checkpoint(self.net, init_weights_path)

            loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            opt = self.build_optim(config["optimizer"], config["lr"])
            network = nn.WithLossCell(self.net, loss)
            network = nn.TrainOneStepCell(network, opt)

            batch_size = config["batch_size"]
            assert batch_size in [8, 16, 32]

            train_set = self.train_set_dict[batch_size]
            train_set_iter = self.train_set_iter_dict[batch_size]

            max_acc = 0

            for epoch in range(num_epoch):
                bar = tqdm(train_set_iter, total=train_set.get_dataset_size(), ncols=100)

                for idx, dic in enumerate(bar):
                    input_img = dic['image']
                    loss = network(input_img, dic['label'])
                    if self.opt.WANDB.OPEN:
                        wandb.log({"loss": loss.asnumpy()})
                    bar.set_description_str(
                        "training: epcoh:{}/{}, idx:{}/{}, loss:{:.6f}".format(epoch + 1, num_epoch, idx + 1,
                                                                               train_set.get_dataset_size(),
                                                                               loss.asnumpy()))

                train_acc = self.eval(self.eval_train_set_iter)
                test_acc = self.eval(self.eval_test_iter)

                acc = {"epoch": epoch + 1, "train acc": train_acc, "test acc": test_acc}
                print(acc)

                if test_acc > max_acc:
                    max_acc = test_acc
                    print("max test acc: ", max_acc)
                    if max_acc > self.global_max_acc:
                        self.global_max_acc = max_acc
                        print("global max test acc: ", self.global_max_acc)
                        self.save_checkpoints()

                if self.opt.WANDB.OPEN:
                    wandb.log(acc)
            if self.opt.WANDB.OPEN:
                wandb.log({"max test acc": max_acc})

    def sweep(self):
        with open(self.opt.WANDB.SWEEP_CONFIG, encoding="utf-8") as f:
            self.sweep_config = json.load(f)
        f.close()
        sweep_id = wandb.sweep(self.sweep_config["inceptionv3"]["sweep_config"], project=self.opt.WANDB.PROJECT_NAME)
        wandb.agent(sweep_id, self.run_sweep)

    def eval(self, data_iter):
        y_test, y_pred = [], []
        for idx, dic in enumerate(data_iter):
            input_img = dic['image']
            output = self.net(input_img)
            predict = np.argmax(output.asnumpy(), axis=1)
            y_test += list(dic['label'].asnumpy())
            y_pred += list(predict)

        test_acc = accuracy_score(y_pred, y_test)
        return test_acc

    def build_optim(self, optim, lr):

        optimizer = None

        if optim == "sgd":
            optimizer = nn.SGD(self.net.trainable_params(), lr)
        elif optim == "adam":
            optimizer = nn.Adam(self.net.trainable_params(), lr)
        elif optim == "adagrad":
            optimizer = nn.Adagrad(self.net.trainable_params(), lr)
        elif optim == "momentum":
            optimizer = nn.Momentum(self.net.trainable_params(), lr, momentum=0.9)

        return optimizer

    def save_checkpoints(self):
        save_checkpoint(self.net, os.path.join(self.opt.TRAIN.SAVE_PATH, self.model_name + '_best_param.ckpt'))
        save_checkpoint(self.net.backbone, os.path.join(self.opt.TRAIN.SAVE_PATH, self.model_name + '_best_param_backbone.ckpt'))
        print("saving param...")

