import math
from PIL import Image, ImageFilter
from logger import logger
from img_utils import alter_img_colour_palette
from curriculm_utils import calculate_available_colours
import img_utils

class ColourCurriculumTransform(object):
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        self.epoch = 0
        self.__update_available_colours()

    def __call__(self, img):
        altered_img = alter_img_colour_palette(img, self.available_colours)
        return altered_img 

    def __update_available_colours(self):
        self.available_colours = calculate_available_colours(self.epoch, self.parameters['t_g'], self.parameters['c_0'], 256)
        logger.info('curriculum (colour): available colours set to {}'.format(self.available_colours)) 

    def set_available_colours(self, num_colours):
        self.available_colours = num_colours

    def get_available_colours(self):
        return self.available_colours

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.__update_available_colours()

class SobelTransform(object):
    def __call__(self, img):
        altered_img = img.convert("L")
        altered_img = img.filter(ImageFilter.FIND_EDGES)

        return altered_img 
