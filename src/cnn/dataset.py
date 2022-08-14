import torch
from torchvision.datasets import CIFAR100, ImageFolder
import torchvision.transforms as tt
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2

class AugmentedDataset(torch.utils.data.Dataset):
  
    def __init__(self, datasets, transform=None):
        combined_list = []
        for i in range(len(datasets)):
            combined_list.extend(self.__return_all_items(datasets[i]))
        self.combined_list = combined_list
    
    def __len__(self):
        return len(self.combined_list)
    
    def __getitem__(self, idx):
        sample = self.combined_list[idx]      
        return sample

    def __return_all_items(self, dataset):
        all_items = []
        for i in range(len(dataset)):
            all_items.append(dataset[i])
        return all_items
    
def load_dataset(dataset_name, data_dir, prior_transforms:list, transforms:list, train = True):
    transform = prior_transforms + transforms
    transform = tt.Compose(transform)
    if dataset_name == "cifar100":
        return CIFAR100(root =data_dir, train = train, transform = transform, download = True)
    elif dataset_name == "mpeg400" or dataset_name == "2dshapes":
        data_path = os.path.join(data_dir, dataset_name)
        ds = ImageFolder(root = data_path, transform= transform)
        ds_size = len(ds)
        n_test = int(0.2 * ds_size)  # take ~20% for test

        if train == None: 
            return ds
        elif train == True:
            return torch.utils.data.Subset(ds, range(n_test, ds_size))
        else:
            return torch.utils.data.Subset(ds, range(n_test))
    else:
        raise ValueError("unsupported dataset")

def calculate_dataset_stats(dataset_name, data_dir, image_size):

    transform =  tt.Compose([
        tt.Resize((32, 32)),
        tt.RandomHorizontalFlip(),
        tt.ToTensor(),
    ]) 

    dataset = load_dataset(
        dataset_name = dataset_name,
        data_dir = data_dir,
        transform = transform,
        train = None)

    data_loader = DataLoader(
        dataset,
        128,
        num_workers=4,
        pin_memory=True)

    nimages = 0
    mean = 0.0
    std = 0.0
    for batch, _ in data_loader:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        nimages += batch.size(0)
        mean += batch.mean(2).sum(0) 
        std += batch.std(2).sum(0)

    # Final step
    mean /= nimages
    std /= nimages

    return (tuple(mean.tolist()), tuple(std.tolist()))