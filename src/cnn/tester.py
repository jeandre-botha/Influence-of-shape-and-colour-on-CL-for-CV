import os
import tensorflow as tf
from os.path import exists as file_exists
from datetime import datetime

from logger import logger
from dataset import Dataset

class Tester:
    def __init__(self, model_name, dataset, config):
        self.model_name = model_name
        self.dataset_name = dataset
        self.config = config
        self.model = None
        self.results = None
        self.models_dir = os.path.abspath(os.path.join(self.config['root_path'], 'models'))
        self.results_dir = os.path.abspath(os.path.join(self.config['root_path'], 'results'))
        self.__init_model()

    def __init_model(self):
        model_path = os.path.join(self.models_dir, self.model_name)
        if file_exists(model_path):
            logger.info('Loading model...')
            try:
                self.model = tf.keras.models.load_model(model_path)
                logger.info('Loading model done')
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
            file_name = 'test_{}_result.txt'.format(str(datetime.now().timestamp()))
            model_results_path =  os.path.join(self.results_dir, self.model_name)
            os.makedirs(model_results_path, exist_ok=True)
            result_path = os.path.join(model_results_path, file_name)
        elif not file_exists(os.path.dirname(result_path)):
            raise ValueError("specified path does not exist")

        with open(result_path, 'w') as result_file:
            result_file.write('test loss: {},  test acc: {}'.format(self.results[0], self.results[1]))

        logger.info('Test results have been saved to "{}"'.format(result_path))

    def test(self):
        logger.info('Loading test data...')
        dataset = Dataset(self.dataset_name)
        test_x = dataset.get_test_data()
        test_y = dataset.get_test_labels()

        test_loader = Dataloader(test_x, test_y, self.config, "test")

        logger.info('Loading test data done')

        logger.info('Evaluating model...')
        self.results = self.model.evaluate(test_loader, batch_size=self.config['batch_size'])
        logger.info('Evaluating model done')

        print(
            "Loss: {:.3f}, Accuracy: {:.3%}".format(
                self.results[0],
                self.results[1]
            )
        )

        self.save_results()
        