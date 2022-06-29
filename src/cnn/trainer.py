import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import matplotlib
import matplotlib.pyplot as plt
from os.path import exists as file_exists

from model import ResNet9
from logger import logger
from utils import get_default_device, to_device
from data_loader import DeviceDataLoader

matplotlib.rcParams['figure.facecolor'] = '#ffffff'


class Trainer:
    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }

    def __init__(self, model_name, dataset, config):
        self.model_name = model_name
        self.dataset_name = dataset
        self.config = config
        self.model = None
        self.history = None
        self.models_dir = os.path.abspath(os.path.join(self.config['root_path'], 'models'))
        self.results_dir = os.path.abspath(os.path.join(self.config['root_path'], 'results'))
        self.data_dir = os.path.abspath(os.path.join(self.config['root_path'], 'data'))

        if self.config['optimizer'] not in self.optimizers:
            raise ValueError('invalid optimizer')

        self.__init_data()
        self.__init_model()


    def __init_data(self):
        logger.info('Loading training data...')
        stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))      #cifar100

        train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                                tt.RandomHorizontalFlip(), 
                                # tt.RandomRotate
                                # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)), 
                                # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                tt.ToTensor(), 
                                tt.Normalize(*stats,inplace=True)])
        valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

        train_ds = CIFAR100(root = self.data_dir, download = True, transform = train_tfms)
        valid_ds = CIFAR100(root = self.data_dir, train = False, transform = valid_tfms)

        train_dl = DataLoader(train_ds, self.config['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
        valid_dl = DataLoader(valid_ds, self.config['batch_size']*2, num_workers=3, pin_memory=True)

        device = get_default_device()
        self.train_dl = DeviceDataLoader(train_dl, device)
        self.valid_dl = DeviceDataLoader(valid_dl, device)
        logger.info('Loading training data done')

    
    def __init_model(self):
        model =  None
        model_path = os.path.join(self.models_dir, self.model_name)
        if file_exists(model_path):
            try:
                logger.info('Loading existing model...')
                model = ResNet9(3, 100)
                model.load_state_dict(torch.load(model_path))
                logger.info('Loading existing model done')
            except:
                logger.warning('could not load model at "{}"'.format(model_path))

        if model == None:
            logger.info('Initializing new model...')
            model = ResNet9(3, 100)

        device = get_default_device()
        torch.cuda.empty_cache()
        self.model = to_device(model, device)  

    def save_model(self, model, path):
        logger.info('Saving model')
        torch.save(model.state_dict(), path)
        logger.info('Model saved to {}'.format(path))

    def test(self):
        @torch.no_grad()
        def evaluate(model, val_loader):
            model.eval()      
            outputs = [model.validation_step(batch) for batch in val_loader]
            return model.validation_epoch_end(outputs)

        evaluate(self.model, self.valid_dl)

    def train(self):
        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        def evaluate(model, val_loader):
            model.eval()      
            outputs = [model.validation_step(batch) for batch in val_loader]
            return model.validation_epoch_end(outputs)
                
        def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                        weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):

            torch.cuda.empty_cache()
            history = []
            
            # Set up cutom optimizer with weight decay
            optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

            # Set up one-cycle learning rate scheduler
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr,
                epochs=epochs, 
                steps_per_epoch=len(train_loader)
            )
            
            for epoch in range(epochs):
                # Training Phase 
                model.train()
                train_losses = []
                lrs = []
                for batch in train_loader:
                    loss = model.training_step(batch)
                    train_losses.append(loss)
                    loss.backward()
                    
                    # Gradient clipping
                    if grad_clip: 
                        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Record & update learning rate
                    lrs.append(get_lr(optimizer))
                    sched.step()
                
                # Validation phase
                result = evaluate(model, val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                result['lrs'] = lrs
                model.epoch_end(epoch, result)
                history.append(result)
            return history

        logger.info('Training model...')
        torch.cuda.empty_cache()
        history = [evaluate(self.model, self.valid_dl)]
        history += fit_one_cycle(
            self.config['epochs'],
            self.config['max_lr'],
            self.model,
            self.train_dl,
            self.valid_dl, 
            grad_clip=self.config['grad_clip'], 
            weight_decay=self.config['weight_decay'], 
            opt_func= self.optimizers[self.config['optimizer']]
        )

        logger.info('Training model done')

        self.save_model(self.model, os.path.join(self.models_dir, self.model_name))