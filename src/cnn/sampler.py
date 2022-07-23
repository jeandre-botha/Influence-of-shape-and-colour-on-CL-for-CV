from torch.utils.data import Sampler
from img_utils import calculate_mean_si, convert_img_to_grayscale, pil_to_skimage, skimage_to_pil

class ComplexityCurriculumSampler(Sampler):
    # want inputs to be an array
    def __init__(self, inputs):
        ind_n_difficulty = []
        for i, p in enumerate(inputs):
            img_data = pil_to_skimage(p[0])
            img_data = convert_img_to_grayscale(img_data)
            ind_n_difficulty.append((i, calculate_mean_si(img_data)))
        self.ind_n_difficulty = ind_n_difficulty
        self.sorted_list = self.__sort_inputs()

    def __sort_inputs(self):
        return sorted(self.ind_n_difficulty, key=lambda x: x[1])

    def __len__(self):
        return len(self.ind_n_difficulty)

    def __iter__(self):
        self.sorted_list = self.__sort_inputs()
        for el in self.sorted_list:
            yield el[0]