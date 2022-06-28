from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

from logger import logger

class Dataset:

    dataset_map = {
        "cifar100": cifar100
    }

    def __init__(self, dataset_name):
        self.__load_dataset(dataset_name)


    def __load_dataset(self, dataset_name):
        if dataset_name not in self.dataset_map:
            raise ValueError('invalid dataset: "{}"'.format(dataset_name))

        # load dataset
        # tf guarantees that the splits are deterministic
        (x_train, y_train), (x_test, y_test) = self.dataset_map[dataset_name].load_data()

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_train_data(self):        
        return self.x_train

    def get_test_data(self):        
        return self.x_test

    def get_train_labels(self, one_hot_encode = True):
        train_labels = self.y_train
        if one_hot_encode:
            train_labels = to_categorical(train_labels)
        
        return train_labels

    def get_test_labels(self, one_hot_encode = True):
        test_labels = self.y_test
        if one_hot_encode:
            test_labels = to_categorical(test_labels)
        
        return test_labels

