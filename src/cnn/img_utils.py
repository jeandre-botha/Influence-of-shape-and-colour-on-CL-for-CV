import skimage
from skimage.color import rgb2gray
from PIL import Image
import math

def __square_root_filter(image_data):
    transformed_img = []
    for row_data in image_data:
        transformed_img_row = []
        for pixel in  row_data:
            transformed_img_row.append(round(255 * math.sqrt(pixel / 255)))
        transformed_img.append(transformed_img_row)
    
    return transformed_img

def calculate_mean_si(image_data):
    tmp_image_data = image_data
    if(len(tmp_image_data.shape)>2):
        tmp_image_data = rgb2gray(tmp_image_data)
        
    sobel_v = skimage.filters.sobel_v(tmp_image_data)
    sobel_h = skimage.filters.sobel_h(tmp_image_data)

    si_r = pow(sobel_v,2) + pow(sobel_h,2)
    si_r = __square_root_filter(si_r)

    total = 0.0
    num_pixels = 0.0
    for row_data in si_r:
        for pixel in  row_data:
            total += pow(pixel, 2)
            num_pixels += 1
    
    return total/num_pixels


def convert_img_to_grayscale(img_data):
    return rgb2gray(img_data)

def alter_img_colour_palette(img_data, num_colours):
    altered_img = img_data.convert('P', palette=Image.ADAPTIVE, colors=num_colours)
    altered_img = altered_img.convert('RGB', palette=Image.ADAPTIVE, colors=num_colours)
    return altered_img
