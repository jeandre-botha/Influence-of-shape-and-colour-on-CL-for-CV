import os
import torch
from datetime import datetime
import torch.nn as nn
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import matplotlib
import matplotlib.pyplot as plt
from os.path import exists as file_exists
import torch.optim as optim

from model import resnet50, eval_training
from logger import logger
from utils import get_default_device, to_device
from data_loader import DeviceDataLoader
from transform import Curriculum
import json

import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

matplotlib.rcParams['figure.facecolor'] = '#ffffff'


class Trainer:
    def __init__(self, model_name, dataset, config):
        self.model_name = model_name
        self.dataset_name = dataset
        self.config = config
        self.model = None
        self.history = None
        self.models_dir = os.path.abspath(os.path.join(self.config['root_path'], 'models'))
        self.results_dir = os.path.abspath(os.path.join(self.config['root_path'], 'results'))
        self.data_dir = os.path.abspath(os.path.join(self.config['root_path'], 'data'))

        self.__init_data()
        self.__init_model()


    def __init_data(self):
        logger.info('Loading training data...')
        stats = ((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))      #cifar100

        train_tfms = [
            tt.RandomCrop(32, padding=4, padding_mode='reflect'),
            tt.RandomHorizontalFlip(),
            # tt.RandomRotate
            # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            tt.ToTensor(),
            tt.Normalize(*stats,inplace=True)
        ]

        self.curriculum_tfm = None
        if 'curriculum' in self.config:
            logger.info('Initiating "{}" curriculum'.format(self.config['curriculum']['name']))
            self.curriculum_tfm = Curriculum(self.config['curriculum']['name'], self.config['curriculum']['parameters'])
            train_tfms.insert(0, self.curriculum_tfm)

        train_tfms = tt.Compose(train_tfms)
        valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

        train_ds = CIFAR100(root = self.data_dir, download = True, transform = train_tfms)
        valid_ds = CIFAR100(root = self.data_dir, train = False, transform = valid_tfms)

        train_dl = DataLoader(train_ds, self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        valid_dl = DataLoader(valid_ds, self.config['batch_size']*2, num_workers=4, pin_memory=True)

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
                model = resnet50()
                model.load_state_dict(torch.load(model_path))
                logger.info('Loading existing model done')
            except:
                logger.warning('could not load model at "{}"'.format(model_path))

        if model == None:
            logger.info('Initializing new model...')
            model = resnet50()

        device = get_default_device()
        torch.cuda.empty_cache()
        self.model = to_device(model, device)

    def save_model(self, model, path):
        logger.info('Saving model')
        torch.save(model.state_dict(), path)
        logger.info('Model saved to {}'.format(path))

    def test(self):
        torch.cuda.empty_cache()
        result = eval_training(self.model, self.valid_dl)
        print("val_loss: {}, val_acc: {}".format(result['val_loss'], result['val_acc']))

        if result_path == None:
            file_name = 'test_{}_result.txt'.format(str(datetime.now().timestamp()))
            model_results_path =  os.path.join(self.results_dir, self.model_name)
            os.makedirs(model_results_path, exist_ok=True)
            result_path = os.path.join(model_results_path, file_name)
        elif not file_exists(os.path.dirname(result_path)):
            raise ValueError("specified path does not exist")

        with open(result_path, 'w') as result_file:
            result_file.write('test loss: {},  test acc: {}'.format(result['val_loss'], result['val_acc']))

        logger.info('Test results have been saved to "{}"'.format(result_path))

    def train(self):
        def fit(epochs, model, train_loader, val_loader, opt_func, train_scheduler=None, grad_clip=None):
            best_acc = -1
            history = []
            if self.curriculum_tfm != None:
                    self.curriculum_tfm.reset_epoch()

            for epoch in range(epochs):
                if train_scheduler != None:
                    train_scheduler.step()
                # Training Phase
                model.train()
                train_losses = []
                train_acc = []
                for batch in train_loader:

                    # opt_func.zero_grad()
                    for param in model.parameters():
                        param.grad = None
                        
                    loss, acc = model.training_step(batch)
                    train_losses.append(loss)
                    train_acc.append(acc)
                    loss.backward()

                    if grad_clip != None:
                        nn.utils.clip_grad_value_(model.parameters(), 0.1)

                    opt_func.step()
                # Validation phase
                result = eval_training(model, val_loader)
                result['train_loss'] = torch.stack(train_losses).mean().item()
                result['train_acc'] = torch.stack(train_acc).mean().item()
                model.epoch_end(epoch, result)
                history.append(result)

                if self.curriculum_tfm != None:
                    self.curriculum_tfm.advance_epoch()


                if epoch > self.config['save_epoch'] and best_acc < result['val_acc']:
                    self.save_model(self.model, os.path.join(self.models_dir, self.model_name))
                    best_acc = result['val_acc']

            return history

        logger.info('Training model...')

        optimizer = torch.optim.SGD
        optimizer_config = self.config['optimizer']
        weight_decay =  self.config['weight_decay']
        if optimizer_config['name'] == 'sgd':
            learning_rate = optimizer_config['learning_rate']
            momentum = optimizer_config['momentum'] if 'momentum' in optimizer_config else 0.0
            use_nesterov = optimizer_config['nesterov'] if 'nesterov' in optimizer_config else False

            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=use_nesterov
            )
        else:
            raise ValueError('Unsupported optimizer {}'.format(self.config['optimizer']['name']))

        grad_clip = self.config['grad_clip'] if 'grad_clip' in self.config else None

        train_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2
        )

        params = {
            "epochs": self.config['epochs'],
            "batch_size": self.config['batch_size'],
            "weight_decay": weight_decay,
            "optimizer": self.config['optimizer']['name'],
            "learning_rate": learning_rate,
            "momentum": momentum,
            "use_nesterov": use_nesterov,
            "train_scheduler_milestones": "[60, 120, 160]",
            "train_scheduler_gamma": "0.2",
        }

        logger.info('Using parameters: {}'.format(json.dumps(params)))
        if 'curriculum' in self.config:
            logger.info('Using curriculum  parameters: {}'.format(json.dumps(self.config['curriculum'])))

        torch.cuda.empty_cache()
        history = [eval_training(self.model, self.valid_dl)]
        history += fit(self.config['epochs'], self.model, self.train_dl, self.valid_dl, optimizer, train_scheduler, grad_clip)

        logger.info('Training model done')

        self.save_training_plots(history)

    def save_training_plots(self, history):
        file_name = 'train_{}_result.png'.format(str(datetime.now().timestamp()))
        model_results_path =  os.path.join(self.results_dir, self.model_name)
        os.makedirs(model_results_path, exist_ok=True)
        result_path = os.path.join(model_results_path, file_name)

        # train_accuracies = [x['train_acc'] for x in history]
        val_accuracies = [x['val_acc'] for x in history]
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]

        plt.subplot(211)
        plt.title('Cross Entropy Loss')
        plt.plot(train_losses, color='blue', label='train')
        plt.plot(val_losses, color='red', label='validation')
        plt.subplot(212)
        plt.title('Classification Accuracy')
        # plt.plot(train_accuracies, color='blue', label='train')
        plt.plot(val_accuracies, color='red', label='validation')
        plt.savefig(result_path)
        logger.info('Training results have been saved to "{}"'.format(result_path))
        plt.close()
