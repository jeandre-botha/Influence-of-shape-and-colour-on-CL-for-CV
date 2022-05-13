import os
import tensorflow as tf
import numpy as np

import utils
from logger import logger

tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Trainer:
    def __init__(self, model_path, data_path, config):
        self.model_path = model_path
        self.data_path = data_path
        self.config = config
    
    def train(self):
        logger.info("Fetching training data")
        metadata_path = os.path.join(self.data_path, 'meta')
        metadata = utils.unpickle(metadata_path)
        superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))

        data_train_path = os.path.join(self.data_path, 'train')
        data_train_dict = utils.unpickle(data_train_path)

        data_train = data_train_dict[b'data']
        label_train = np.array(data_train_dict[b'fine_labels'])

        logger.info("Initializing model")

        model = tf.keras.applications.resnet50.ResNet50(
            include_top=True,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
        )

        model.compile(
            optimizer='adam',
            loss="categorical_crossentropy",
            metrics=['accuracy']
        )
