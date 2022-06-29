from utils import to_device

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# if 'curriculum' in self.config and self.config['curriculum']['name'] == 'colour':
#     parameters =  self.config['curriculum']['parameters']

#     t = self.epochs
#     t_g = parameters['t_g']
#     c_0 = parameters['c_0']
#     c_t = min(1, t*((1-c_0)/t_g)+c_0)
#     total_colours = 256
#     available_colours = math.ceil(c_t*total_colours)            

#     img = tf.keras.preprocessing.image.array_to_img(img)
#     img = img.convert('P', palette=Image.ADAPTIVE, colors=available_colours)
#     img = img.convert('RGB', palette=Image.ADAPTIVE, colors=available_colours)
#     img = tf.keras.preprocessing.image.img_to_array(img)