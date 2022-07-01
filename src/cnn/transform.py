import math
from PIL import Image

class Curriculum(object):
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        self.epoch = 0
        self.available_colours = self.__calculate_available_colours()

    def __call__(self, img):
        if  self.name == 'colour':
            img = img.convert('P', palette=Image.ADAPTIVE, colors=self.available_colours)
            img = img.convert('RGB', palette=Image.ADAPTIVE, colors=self.available_colours)
        return img 

    def __calculate_available_colours(self):
        t = self.epoch
        t_g = self.parameters['t_g']
        c_0 = self.parameters['c_0']
        c_t = min(1, t*((1-c_0)/t_g)+c_0)
        total_colours = 256
        return math.ceil(c_t*total_colours) 

    def advance_epoch(self):
        self.epoch += 1
        self.available_colours = self.__calculate_available_colours()

    def reset_epoch(self):
        self.epoch += 0
        self.available_colours = self.__calculate_available_colours()
