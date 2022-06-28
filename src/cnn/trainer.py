import os
import math   
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import datasets, layers, models, losses, Model
import numpy as np
from matplotlib import pyplot
from datetime import datetime
from os.path import exists as file_exists
from PIL import Image

from logger import logger
from dataset import Dataset

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CustomModel(tf.keras.Model):
    def __init__(self, config, steps_per_epoch, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)
        self.config = config
        self.steps_per_epoch = steps_per_epoch

        self.steps_completed = 0


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        x_aug = self.augment(x)
        
        with tf.GradientTape() as tape:
            y_pred = self(x_aug, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        self.steps_completed += 1
        return {m.name: m.result() for m in self.metrics}


    def augment(self, tensor):
        tensor = tf.cast(x=tensor, dtype=tf.float32)

        # apply curriculum based augmentations
        if 'curriculum' in self.config and self.config['curriculum']['name'] == 'colour':
            parameters =  self.config['curriculum']['parameters']

            t = math.floor(self.steps_completed/self.steps_per_epoch)
            t_g = parameters['t_g']
            c_0 = parameters['c_0']
            c_t = min(1, t*((1-c_0)/t_g)+c_0)
            total_colours = 256
            available_colours = math.ceil(c_t*total_colours)            

            result_images = []
            for t in tensor:
                img = tf.keras.preprocessing.image.array_to_img(t)
                img = img.convert('P', palette=Image.ADAPTIVE, colors=available_colours)
                img = img.convert('RGB', palette=Image.ADAPTIVE, colors=available_colours)
                result_images.append(tf.keras.preprocessing.image.img_to_array(img))

            tensor = np.array(result_images)

        # apply remaining augmentations
        tensor = tf.divide(x=tensor, y=tf.constant(255.))
        tensor = tf.image.random_flip_left_right(image=tensor)
        # tensor = tf.image.random_brightness(image=tensor, max_delta=2e-1)
        return tensor

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
        self.train_x = self.dataset.get_train_data(normalize=False)
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

         # create base ResNet model
        base_model = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=True,
            weights = None,
            input_tensor=None,
            input_shape=(32, 32, 3),
            pooling=None,
            classes=100,
            classifier_activation='softmax'
        )

        # add top (dense) layer
        # x = layers.Flatten()(base_model.output)
        # x = layers.Dense(100, activation='relu')(x)
        # predictions = layers.Dense(100, activation = 'softmax')(x)

        steps_per_epoch = math.ceil((len(self.train_x) / self.config['batch_size']))

        # build final model
        model = CustomModel(
            inputs = base_model.input,
            outputs = base_model.output,
            config = self.config,
            steps_per_epoch = steps_per_epoch)

        self.model = model

        learning_rate = self.config['learning_rate']
        weight_decay = self.config['weight_decay']
        lr_decay = self.config['lr_decay'] if 'lr_decay' in self.config else None       

        lr_schedule = learning_rate
        # if lr_decay != None:
        #     logger.debug("Adding exponential decay lr schedule")
        #     decay_rate = lr_decay['decay_rate']
        #     decay_epochs = lr_decay['decay_epochs']

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
        logger.info('Config: ', self.config)


        # Prepare the training dataset.
        batch_size = self.config['batch_size']

        validation_size = math.floor(len(self.train_x)*0.1)

        train_x = self.train_x[:-validation_size]
        train_y = self.train_y[:-validation_size]
        val_x  = self.train_x[-validation_size:]
        val_y  = self.train_y[-validation_size:]

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)


        def MultiStepLrScheduler(epoch, lr):
            if epoch == 60 or epoch == 120 or epoch == 160:
                return lr/5
            else:
                return lr

        schedulerCallback = tf.keras.callbacks.LearningRateScheduler(MultiStepLrScheduler)

        history = self.model.fit(
            train_dataset,
            batch_size = batch_size,
            epochs = self.config['epochs'],
            validation_data=(val_x, val_y),
            callbacks=[schedulerCallback])

        self.history = history

        logger.info('Training model done')

        self.save_model()
        self.save_summary()
        self.test()


    def test(self):
        logger.info('Loading test data...')
        dataset = Dataset(self.dataset_name)
        test_x = dataset.get_test_data(normalize=True)
        test_y = dataset.get_test_labels()
        logger.info('Loading test data done')

        logger.info('Evaluating model...')
        self.results = self.model.evaluate(test_x, test_y, batch_size=self.config['batch_size'])
        logger.info('Evaluating model done')

        print(
            "Loss: {:.3f}, Accuracy: {:.3%}".format(
                self.results[0],
                self.results[1]
            )
        )