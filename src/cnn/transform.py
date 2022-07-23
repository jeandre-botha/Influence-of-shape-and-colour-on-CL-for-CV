import math
from PIL import Image
from logger import logger
from img_utils import alter_img_colour_palette
from curriculm_utils import calculate_available_colours

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
        self.available_colours = calculate_available_colours(self.epoch, self.parameters['t_g'], self.parameters['c_0'], 256)
        logger.info('curriculum (colour): available colours set to {}'.format(self.available_colours)) 

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.__update_available_colours()
