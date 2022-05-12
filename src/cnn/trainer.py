import tensorflow as tf

class Trainer:
    def __init__(self, model_path, data_path, config):
        self.model_path = data_path
        self.data_path = model_path
        self.config = config
        raise NotImplementedError()
