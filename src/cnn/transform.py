import math
from PIL import Image
from logger import logger
from img_utils import alter_img_colour_palette

class ColourCurriculumTransform(object):
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        self.epoch = 0
        self.__update_available_colours()

    def __call__(self, img):
        if self.name == 'colour':
            img = alter_img_colour_palette(img, self.available_colours)
        return img 

    def __update_available_colours(self):
        t = self.epoch
        t_g = self.parameters['t_g']
        c_0 = self.parameters['c_0']
        c_t = min(1, t*((1-c_0)/t_g)+c_0)
        total_colours = 256
        self.available_colours = math.ceil(c_t*total_colours)
        logger.info('curriculum (colour): available colours set to {}'.format(self.available_colours)) 

    def advance_epoch(self):
        self.epoch += 1
        self.__update_available_colours()

    def reset_epoch(self):
        self.epoch = 0
        self.__update_available_colours()
