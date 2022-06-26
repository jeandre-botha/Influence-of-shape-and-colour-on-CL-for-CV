import tensorflow as tf

def get_optimizer(config:dict):
    return tf.keras.optimizers.SGD(
                    learning_rate=config['learning_rate'],
                    momentum=config['momentum'])

def get_loss_fn(config:dict):
    return tf.keras.losses.categorical_crossentropy