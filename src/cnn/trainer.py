import os
import math
import tensorflow as tf
import tensorflow_addons as tfa
from matplotlib import pyplot
from datetime import datetime
from os.path import exists as file_exists

from logger import logger
from dataset import Dataset
from Dataloader import Dataloader
from lr_schedule import resolve_schedular_callback

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
        self.models_dir = os.path.abspath(os.path.join(self.config['root_path'], 'models'))
        self.results_dir = os.path.abspath(os.path.join(self.config['root_path'], 'results'))

        self.train_x = None
        self.train_y = None
        self.__init_data()
        self.__init_model()

    def __init_data(self):
        logger.info('Loading training data...')
        self.dataset = Dataset(self.dataset_name)
        self.train_x = self.dataset.get_train_data()
        self.train_y = self.dataset.get_train_labels()
        logger.info('Loading training data done')

    
    def __init_model(self):
        model_path = os.path.join(self.models_dir, self.model_name)
        if file_exists(model_path):
            try:
                logger.info('Loading existing model...')
                self.model = tf.keras.models.load_model(model_path)
                logger.info('Loading existing model done')
                return
            except:
                logger.warning('could not load model at "{}"'.format(model_path))

        logger.info('Initializing new model...')

        self.model = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=True,
            weights = None,
            input_tensor=None,
            input_shape=(32, 32, 3),
            pooling=None,
            classes=100,
            classifier_activation='softmax'
        )

        learning_rate = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        lr_decay = self.config['lr_decay'] if 'lr_decay' in self.config else None       

        lr_schedule = learning_rate
        # if lr_decay != None:
        #     logger.debug("Adding exponential decay lr schedule")
        #     decay_rate = lr_decay['decay_rate']
        #     decay_epochs = lr_decay['decay_epochs']
        #     steps_per_epoch = math.ceil((len(self.train_x) / self.config['batch_size']))
        #     decay_steps = decay_epochs * steps_per_epoch

        #     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #         learning_rate,
        #         decay_steps=decay_steps,
        #         decay_rate=decay_rate,
        #         staircase=False
        #     )

        optimizer = tfa.optimizers.SGDW(
            weight_decay = weight_decay,
            learning_rate = lr_schedule,
            nesterov = self.config['nesterov'],
            momentum = self.config['momentum'],
        )

        optimizer = tf.keras.optimizers.SGD(
            learning_rate = lr_schedule,
            nesterov = self.config['nesterov'],
            momentum = self.config['momentum'],
        )

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']
        )

        logger.info('Initializing new model done')

    def save_summary(self, result_path = None):
        if self.history == None:
            logger.error('no available training history found, please run train first')
            return

        if result_path == None:
            file_name = 'train_{}_result.png'.format(str(datetime.now().timestamp()))
            model_results_path =  os.path.join(self.results_dir, self.model_name)
            os.makedirs(model_results_path, exist_ok=True)
            result_path = os.path.join(model_results_path, file_name)
        elif not file_exists(os.path.dirname(result_path)):
            raise ValueError("specified path does not exist")

        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(self.history.history['loss'], color='blue', label='train')
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(self.history.history['accuracy'], color='blue', label='train')
        pyplot.savefig(result_path)
        logger.info('Training results have been saved to "{}"'.format(result_path))
        pyplot.close()

    def save_model(self, model_path = None):
        if model_path == None:
            model_path = os.path.join(self.models_dir, self.model_name)

        logger.info('Saving model...')

        self.model.save(model_path)
        logger.info('Model saved to destination: "{}"'.format(model_path))

    def train(self):
        logger.info('Training model...')


        # Prepare the training dataset.
        batch_size = self.config['batch_size']

        validation_size = math.floor(len(self.train_x)*0.1)

        train_x = self.train_x[:-validation_size]
        train_y = self.train_y[:-validation_size]
        val_x  = self.train_x[-validation_size:]
        val_y  = self.train_y[-validation_size:]

        train_loader = Dataloader(train_x, train_y, self.config, "train")
        validation_loader = Dataloader(val_x, val_y, self.config, "test")

        history = self.model.fit(
            train_loader,
            batch_size = batch_size,
            epochs = self.config['epochs'],
            validation_data=validation_loader,
            callbacks=[resolve_schedular_callback('reduce_on_plateau')])

        self.history = history

        logger.info('Training model done')

        self.save_model()
        self.save_summary()
        self.test()


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