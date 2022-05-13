import os
import tensorflow as tf
from os.path import exists as file_exists
from datetime import datetime

from logger import logger
from constants import models_dir, results_dir
from dataset import Dataset

class Tester:
    def __init__(self, model_name, dataset, config):
        self.model_name = model_name
        self.dataset_name = dataset
        self.config = config
        self.model = None
        self.results = None
        self.__init_model()

    def __init_model(self):
        model_path = os.path.join(models_dir, self.model_name)
        if file_exists(model_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                return
            except:
               raise ValueError('could not load model at "{}"'.format(model_path))
        else:
            raise ValueError('no model found at "{}"'.format(model_path))

    def save_results(self, result_path = None):
        if self.results == None:
            logger.error('no available results found, please run test first')
            return

        if result_path == None:
            file_name = '{}_test_{}_result.txt'.format(self.model_name, str(datetime.now().timestamp()))
            result_path = os.path.join(results_dir, file_name)

        with open(result_path, 'w') as result_file:
            result_file.write('test loss: {},  test acc: {}'.format(self.results[0], self.results[1]))

    def test(self):
        logger.info('Loading test data...')
        dataset = Dataset(self.dataset_name)
        test_x = dataset.get_test_data()
        test_y = dataset.get_test_labels()
        logger.info('Loading test data done')

        logger.info('Evaluating model...')
        self.results = self.model.evaluate(test_x, test_y, batch_size=self.config['batch_size'])
        logger.info('Evaluating model done')
        logger.info('Evaluation results: test loss: {},  test acc: {}'.format(self.results[0], self.results[1]))

        self.save_results()