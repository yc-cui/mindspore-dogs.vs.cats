import wandb
import json
import numpy as np
from sklearn.kernel_approximation import RBFSampler, Nystroem
from utils.model_utils import setup_seed
from sklearn.decomposition import IncrementalPCA
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import os
import glob

setup_seed(22)


class SVMModel:
    def __init__(self, opt):
        self.opt = opt
        self.model_name = "model_{}".format(self.opt.MODEL.NAME)

    def run_sweep(self):
        setup_seed(22)
        self.opt.WANDB.LOG_DIR = os.path.join("./logs/", self.model_name)
        with wandb.init(name=self.model_name,
                        config=self.opt,
                        notes=self.opt.WANDB.LOG_DIR,
                        resume=self.opt.WANDB.RESUME,
                        ) as run:
            config = wandb.config

            wandb.run.name = "_".join([self.model_name, config["input"], "n_comp", str(config["n_components"]),
                                       "kernel", config["kernel"], "gamma", str(round(config["gamma"], 4)),
                                       "C", str(round(config["C"], 4))])

            self.sampler = self.build_kernel(config["kernel"], config["gamma"])
            self.svm_sgd = linear_model.SGDClassifier(random_state=22, alpha=config["C"])
            self.ipca = IncrementalPCA(n_components=config["n_components"], batch_size=128)

            train_features_list, train_labels_list = self.build_features(config["input"], train=True)
            test_features_list, test_labels_list = self.build_features(config["input"], train=False)

            for feature_path in train_features_list:
                feature_np = np.load(feature_path)
                feature_np = self.build_input(config["input"], feature_np)
                if feature_np.shape[0] < config["n_components"]:
                    print(f"find batch {feature_np.shape[0]} < n_components {config['n_components']}, continue...")
                    continue
                self.ipca.partial_fit(feature_np)
                print(f"done pca partial fit: {feature_path}")

            for feature_path, label_path in zip(train_features_list, train_labels_list):
                feature_np = np.load(feature_path)
                label_np = np.load(label_path)
                feature_np = self.build_input(config["input"], feature_np)
                feature_transform = self.ipca.transform(feature_np)
                if feature_transform.shape[0] < config["n_components"]:
                    print(f"find batch {feature_np.shape[0]} < n_components {config['n_components']}, continue...")
                    continue
                feature_map = feature_transform if self.sampler is None else self.sampler.fit_transform(feature_transform)
                self.svm_sgd.partial_fit(feature_map, label_np, classes=[0, 1])
                print(f"done svm partial fit: {feature_path}")

            self.config = config
            train_acc = self.eval(train_features_list, train_labels_list)
            test_acc = self.eval(test_features_list, test_labels_list)
            acc = {"train acc": train_acc, "test acc": test_acc}
            print(wandb.run.name)
            print(acc)
            wandb.log(acc)
            wandb.log({"max test acc": test_acc})

    def sweep(self):
        with open(self.opt.WANDB.SWEEP_CONFIG, encoding="utf-8") as f:
            self.sweep_config = json.load(f)
        f.close()
        sweep_id = wandb.sweep(self.sweep_config["svm"]["sweep_config"], project=self.opt.WANDB.PROJECT_NAME)
        wandb.agent(sweep_id, self.run_sweep)

    def build_kernel(self, kernel, gamma):

        sampler = None

        if kernel == "RBF":
            sampler = RBFSampler(gamma=gamma, random_state=22)
        elif kernel == "Nystroem":
            sampler = Nystroem(gamma=gamma, random_state=22)

        return sampler

    def build_input(self, input, feature_np):

        if input == "origin":
            batch = feature_np.shape[0]
            feature_np = feature_np.reshape(batch, -1)
        else:
            feature_np = np.mean(feature_np, (2, 3))

        return feature_np


    def build_features(self, input, train):

        train = "train" if train else "test"
        features_path = f"./checkpoints/features/{train}/{input}_128_features"
        features_list = glob.glob(os.path.join(features_path, "feature*"))
        labels_list = glob.glob(os.path.join(features_path, "label*"))

        return features_list, labels_list

    def eval(self, features_list, labels_list):
        print("evaluating...")
        y_true = np.ndarray([], dtype=int)
        y_pred = np.ndarray([], dtype=int)
        for feature_path, label_path in zip(features_list, labels_list):
            feature_np = np.load(feature_path)
            label_np = np.load(label_path)
            y_true = np.append(y_true, label_np)
            feature_np = self.build_input(self.config["input"], feature_np)
            feature_transform = self.ipca.transform(feature_np)
            feature_map = feature_transform if self.sampler is None else self.sampler.transform(feature_transform)
            y_pred = np.append(y_pred, self.svm_sgd.predict(feature_map))

        return accuracy_score(y_true, y_pred)
