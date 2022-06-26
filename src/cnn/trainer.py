import os
import math   
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, Model
import numpy as np
from matplotlib import pyplot
from datetime import datetime
from os.path import exists as file_exists
from PIL import Image

from logger import logger
from dataset import Dataset
from constants import results_dir, models_dir
import model_utils

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
                logger.info('Loading existing model...')
                self.model = tf.keras.models.load_model(model_path)
                logger.info('Loading existing model done')
                return
            except:
                logger.warning('could not load model at "{}"'.format(model_path))

        logger.info('Initializing new model...')

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

        self.model = model
        self.loss_fn = tf.keras.losses.categorical_crossentropy
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.config['learning_rate'],
            momentum=self.config['momentum']
        )
        logger.info('Initializing new model done')

    def save_summary(self, result_path = None):
        if self.history == None:
            logger.error('no available training history found, please run train first')
            return

        if result_path == None:
            file_name = 'train_{}_result.png'.format(str(datetime.now().timestamp()))
            model_results_path =  os.path.join(results_dir, self.model_name)
            os.makedirs(model_results_path, exist_ok=True)
            result_path = os.path.join(model_results_path, file_name)
        elif not file_exists(os.path.dirname(result_path)):
            raise ValueError("specified path does not exist")

        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(self.history['loss'], color='blue', label='train')
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(self.history['accuracy'], color='blue', label='train')
        pyplot.savefig(result_path)
        logger.info('Training results have been saved to "{}"'.format(result_path))
        pyplot.close()

    def save_model(self, model_path = None):
        if model_path == None:
            model_path = os.path.join(models_dir, self.model_name)

        logger.info('Saving model...')
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy']
        )
        self.model.save(model_path)
        logger.info('Model saved to destination: "{}"'.format(model_path))

    def augment(self, tensor, epoch):
        tensor = tf.cast(x=tensor, dtype=tf.float32)

        # apply curriculum based augmentations
        if 'curriculum' in self.config and self.config['curriculum']['name'] == 'colour':
            parameters =  self.config['curriculum']['parameters']

            t = epoch
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
        # tensor = tf.image.random_flip_left_right(image=tensor)
        # tensor = tf.image.random_brightness(image=tensor, max_delta=2e-1)
        # tensor = tf.image.random_crop(value=tensor, size=(64, 64, 1))
        return tensor

    def train(self):
        logger.info('Loading training data...')
        dataset = Dataset(self.dataset_name)
        train_x = dataset.get_train_data()
        train_y = dataset.get_train_labels()
        logger.info('Loading training data done')

        logger.info('Training model...')

        # Prepare the training dataset.
        batch_size = self.config['batch_size']
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        history = {
            "loss": [],
            "accuracy": []
        }

        for epoch in range(self.config['epochs']):
            print("\nStart of epoch %d" % (epoch,))
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # augment training images
                    x_batch_train = self.augment(x_batch_train, epoch)

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = self.model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = self.loss_fn(y_batch_train, logits)
                    epoch_loss_avg.update_state(loss_value)
                    epoch_accuracy.update_state(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

                # Log every 50 batches.
                if step % 50 == 0:
                    print(
                        "Step {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                            step,
                            epoch_loss_avg.result(),
                            epoch_accuracy.result()
                        )
                    )
            history["loss"].append(epoch_loss_avg.result())
            history["accuracy"].append(epoch_accuracy.result())

            print(
                "Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                    epoch,
                    epoch_loss_avg.result(),
                    epoch_accuracy.result()
                )
            )

        self.history = history

        logger.info('Training model done')

        self.save_model()
        self.save_summary()