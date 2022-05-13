import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
import numpy as np
from matplotlib import pyplot
from datetime import datetime
from os.path import exists as file_exists

from logger import logger
from dataset import Dataset
from constants import results_dir, models_dir

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Trainer:
    def __init__(self, model_name, dataset, config):
        self.model_name = model_name
        self.dataset_name = dataset
        self.config = config
        self.model = None
        self.history = None
        self.__init_model()

    
    def __init_model(self):
        model_path = os.path.join(models_dir, self.model_name)
        if file_exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                return
            except:
                logger.warning('could not load model at "{}"'.format(model_path))

        logger.info("Initializing new model")

         # create base ResNet model
        base_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            input_tensor=None,
            input_shape=(32, 32, 3),
            pooling=None,
            classes=1000,
        )

        # add top (dense) layer
        x = layers.Flatten()(base_model.output)
        x = layers.Dense(1000, activation='relu')(x)
        predictions = layers.Dense(100, activation = 'softmax')(x)

        # build final model
        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(optimizer='adam', loss=losses.categorical_crossentropy, metrics=['accuracy'])

        self.model = model

    def display_summary(self):
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(self.history.history['loss'], color='blue', label='train')
        # pyplot.plot(self.history.history['val_loss'], color='orange', label='test')
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(self.history.history['accuracy'], color='blue', label='train')
        # pyplot.plot(self.history.history['val_accuracy'], color='orange', label='test')
        file_name = '{}_plot.png'.format(self.model_name)
        pyplot.savefig(os.path.join(results_dir, file_name))
        
        pyplot.close()

    def train(self):
        logger.info("Loading training data")
        dataset = Dataset(self.dataset_name)
        train_x = dataset.get_train_data()
        train_y = dataset.get_train_labels()

        logger.info("Training model")
        self.history = self.model.fit(
            train_x,
            train_y,
            epochs=1,
            batch_size=64,
            verbose=1
        )

        logger.info("Training complete")

        self.display_summary()