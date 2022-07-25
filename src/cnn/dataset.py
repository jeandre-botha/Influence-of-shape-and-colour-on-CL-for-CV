import torch

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
    