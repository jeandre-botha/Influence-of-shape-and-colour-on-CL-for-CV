import cv2
import math
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from albumentations import Compose, PadIfNeeded, RandomCrop, HorizontalFlip,Normalize

class Dataloader(Sequence):
    def __init__(self, x_set, y_set, config, mode="train", shuffle=True):
        self.x, self.y = x_set, y_set
        self.config = config
        self.batch_size = config['batch_size']
        self.mode = mode
        self.shuffle=shuffle
        self.epochs = 0
        self.on_epoch_end()
    
    def augment(self, img):
        if self.mode == "train":
            img_preprocessor = Compose([
                PadIfNeeded(40, 40, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                RandomCrop(32, 32),
                HorizontalFlip(p=0.5),
                Normalize (mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), max_pixel_value=255.0)
            ])

            if 'curriculum' in self.config and self.config['curriculum']['name'] == 'colour':
                parameters =  self.config['curriculum']['parameters']

                t = self.epochs
                t_g = parameters['t_g']
                c_0 = parameters['c_0']
                c_t = min(1, t*((1-c_0)/t_g)+c_0)
                total_colours = 256
                available_colours = math.ceil(c_t*total_colours)            

                tmp = tf.keras.preprocessing.image.array_to_img(img)
                tmp = tmp.convert('P', palette=Image.ADAPTIVE, colors=available_colours)
                tmp = tmp.convert('RGB', palette=Image.ADAPTIVE, colors=available_colours)
                tmp = tf.keras.preprocessing.image.img_to_array(tmp)

                return img_preprocessor(image=tmp)["image"]

            return img_preprocessor(image=img)["image"]
        else:
            img_preprocessor = Compose([
                Normalize (mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), max_pixel_value=255.0)
            ])
            return img_preprocessor(image=img)["image"]

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.stack([
            self.augment(x) for x in batch_x
        ], axis=0), np.array(batch_y)

    def on_epoch_end(self):
        self.epochs += 1
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)