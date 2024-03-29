import os
import torch
from datetime import datetime
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import matplotlib
import matplotlib.pyplot as plt
from os.path import exists as file_exists
import torch.optim as optim
import time as t
import sys

from model import resnet50, eval_training
from logger import logger
from utils import get_default_device, to_device
from data_loader import DeviceDataLoader
from transform import ColourCurriculumTransform, SobelTransform
from img_utils import calculate_mean_si, convert_img_to_grayscale, pil_to_skimage
from  curriculm_utils import calculate_num_easiest_examples
import json
import torch.nn.functional as F
from timeit import default_timer as timer
from datetime import timedelta
from dataset import AugmentedDataset, load_dataset, calculate_dataset_stats

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

matplotlib.rcParams['figure.facecolor'] = '#ffffff'
class Trainer:
    dataSetNormalizationStats = {
        "cifar100": ((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
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
        self.metadata = {}
        self.training_dl = None
        self.validation_dl = None
        self.warmup_epochs = self.config['warmup_epochs'] if 'warmup_epochs' in self.config else 0
        self.__init_data()
        self.__setup_data_loaders()
        self.__init_model()


    def __init_data(self):
        logger.info('Loading training data...')

        stats = None
        if self.dataset_name in self.dataSetNormalizationStats:
            stats = self.dataSetNormalizationStats[self.dataset_name]
        else:
            logger.info("Stats for dataset unavailable, calculating...")
            stats = calculate_dataset_stats(self.dataset_name, self.data_dir, 32)
            logger.info("Dataset stats calculated.")


        universal_tfms = [tt.Resize((32, 32))]

        train_tfms = [
            tt.RandomCrop(32, padding=4, padding_mode='reflect'),
            tt.RandomHorizontalFlip(),
            # tt.RandomRotate
            # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            tt.ToTensor(),
            tt.Normalize(*stats,inplace=True)
        ]

        validation_tfms = [ tt.ToTensor(), tt.Normalize(*stats)]

        if 'use_gray_scale' in self.config and self.config['use_gray_scale'] == True:
            universal_tfms.append(tt.Grayscale(num_output_channels=3))

        self.curriculum_tfm = None
        if 'curriculum' in self.config and (self.config['curriculum']['name'] == "colour" or self.config['curriculum']['name'] == "colour_alt"):
            logger.info('Initiating "{}" curriculum'.format(self.config['curriculum']['name']))
            self.curriculum_tfm = ColourCurriculumTransform(self.config['curriculum']['name'], self.config['curriculum']['parameters'])
            train_tfms.insert(0, self.curriculum_tfm)


        if 'curriculum' in self.config and self.config['curriculum']['name'] == "complexity":
            logger.info('Initiating "{}" curriculum'.format(self.config['curriculum']['name']))

            self.train_ds = load_dataset(self.dataset_name, self.data_dir, universal_tfms, train_tfms, train=True)

            logger.info('Creating augmented dataset done')
            logger.info('Calculating image difficulty scores...')
            start = timer()

            tmp_ds = load_dataset(self.dataset_name, self.data_dir, universal_tfms, None, train=True)

            ind_n_difficulty = []

            for i in range(len(tmp_ds)):
                img_data = pil_to_skimage(tmp_ds[i][0])
                ind_n_difficulty.append((i, calculate_mean_si(img_data)))
            
            ind_n_difficulty = sorted(ind_n_difficulty, key=lambda x: x[1])

            self.metadata["complexity_ranked_indicies"] = [x[0] for x in ind_n_difficulty]

            end = timer()
            print("ds size: ", len(self.train_ds))

            logger.info('Calculating image difficulty scores done, took {}'.format(timedelta(seconds=end-start)))
        else:
            self.train_ds = load_dataset(self.dataset_name, self.data_dir, universal_tfms, train_tfms, train=True)

        self.validation_ds = load_dataset(self.dataset_name, self.data_dir, universal_tfms,validation_tfms, train=False)

        logger.info('Loading training data done')

    def __setup_data_loaders(self):
        device = get_default_device()

        val_dl = DataLoader(self.validation_ds, self.config['batch_size']*2, num_workers=4, pin_memory=True)
        self.validation_dl = DeviceDataLoader(val_dl, device)

        if  'curriculum' in self.config and self.config['curriculum']['name'] == "complexity":
            curriculum_params = self.config['curriculum']['parameters']
            t_g = curriculum_params['t_g']
            u_0 = curriculum_params['u_0']
            p = curriculum_params['p']
            num_inputs = len(self.train_ds)
            num_easiest_examples = calculate_num_easiest_examples(0, t_g, u_0, p, num_inputs)

            subset_indicies = self.metadata["complexity_ranked_indicies"][0:num_easiest_examples-1]
            subset = torch.utils.data.Subset(self.train_ds, subset_indicies)

            logger.info('Starting training on subset of size {}'.format(num_easiest_examples))

            train_dl = DataLoader(
                subset,
                self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            self.training_dl = DeviceDataLoader(train_dl, device)
        else:
            logger.info('Training on full dataset')
            train_dl = DataLoader(
                self.train_ds,
                self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            self.training_dl = DeviceDataLoader(train_dl, device)

    def __init_model(self):
        model =  None
        model_path = os.path.join(self.models_dir, self.model_name)
        if file_exists(model_path):
            try:
                logger.info('Loading existing model...')
                model = resnet50(self.config['num_classes'])
                model.load_state_dict(torch.load(model_path))
                logger.info('Loading existing model done')
            except:
                logger.warning('could not load model at "{}"'.format(model_path))

        if model == None:
            logger.info('Initializing new model...')
            model = resnet50(self.config['num_classes'])

        if 'replacement_classes' in self.config:
            model.replace_head(self.config['replacement_classes'])

        device = get_default_device()
        torch.cuda.empty_cache()
        self.model = to_device(model, device)

    def begin_epoch(self, epoch):
        if self.curriculum_tfm != None:
            if self.config['curriculum']['name'] == "colour":
                self.curriculum_tfm.set_epoch(epoch)

        if  'curriculum' in self.config and self.config['curriculum']['name'] == "complexity":
            device = get_default_device()

            curriculum_params = self.config['curriculum']['parameters']
            t_g = curriculum_params['t_g']
            u_0 = curriculum_params['u_0']
            p = curriculum_params['p']
            num_inputs = len(self.train_ds)
            num_easiest_examples = calculate_num_easiest_examples(epoch, t_g, u_0, p, num_inputs)

            subset_indicies = self.metadata["complexity_ranked_indicies"][0:num_easiest_examples-1]
            subset = torch.utils.data.Subset(self.train_ds, subset_indicies)

            logger.info('Training subset size changed to {}'.format(num_easiest_examples))

            train_dl = DataLoader(
                subset,
                self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            self.training_dl = DeviceDataLoader(train_dl, device)

    def end_epoch(self, epoch):
        if self.warmup_epochs > 0:
            self.warmup_epochs -= 1

    def save_model(self, model, path):
        logger.info('Saving model')
        torch.save(model.state_dict(), path)
        logger.info('Model saved to {}'.format(path))

    def test(self):
        torch.cuda.empty_cache()
        result = eval_training(self.model, self.validation_dl)
        print("val_loss: {}, val_acc: {}".format(result['val_loss'], result['val_acc']))

        file_name = 'test_{}_result.txt'.format(str(datetime.now().timestamp()))
        model_results_path =  os.path.join(self.results_dir, self.model_name)
        os.makedirs(model_results_path, exist_ok=True)
        result_path = os.path.join(model_results_path, file_name)

        with open(result_path, 'w') as result_file:
            result_file.write('test loss: {},  test acc: {}'.format(result['val_loss'], result['val_acc']))

        logger.info('Test results have been saved to "{}"'.format(result_path))

    def test_convergence(self, history:list, best_loss, best_epoch, current_epoch, patience = 10, min_delta = 0.0001):
        patience = self.config['early_stop_patience'] if 'early_stop_patience' in self.config else 10
        min_delta = self.config['early_stop_min_delta'] if 'early_stop_min_delta' in self.config else 0.0001

        if len(history) < patience or self.warmup_epochs > 0 or  (current_epoch - best_epoch) < patience:
            return False

        count = 0
        for i in range(best_epoch+1, current_epoch):
            if best_loss - history[i]["val_loss"]  < min_delta:
                count += 1
        
        return count >= patience
        

    def fit(self, start_time, epochs, model, opt_func, train_scheduler=None, step_schedule_on_batch= True, grad_clip=None):
        best_loss = sys.float_info.max
        best_epoch = 0
        history = []

        for epoch in range(epochs):
            self.begin_epoch(epoch)

            if (train_scheduler != None) and (not step_schedule_on_batch):
                train_scheduler.step()
            # Training Phase
            model.train()
            train_losses = []
            train_acc = []
            for batch in self.training_dl:

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

                if (train_scheduler != None) and step_schedule_on_batch:
                    train_scheduler.step()
            # Validation phase
            result = eval_training(model, self.validation_dl)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            result['train_acc'] = torch.stack(train_acc).mean().item()
            model.epoch_end(epoch, result, t.time() - start_time)
            history.append(result)

            if 'early_stop_enabled' in self.config and self.config['early_stop_enabled'] == True:
                if self.test_convergence(history, best_loss, best_epoch, epoch) == True:
                    logger.info('Model converged, halting model training')

                    if self.config['curriculum']['name'] == "colour_alt":
                        if self.curriculum_tfm != None:
                            current_available_colours = self.curriculum_tfm.get_available_colours()
                                
                            if current_available_colours == 256:
                                if best_loss > result['val_loss']:
                                    self.save_model(self.model, os.path.join(self.models_dir, self.model_name))
                                    return history
                            else:
                                if current_available_colours < 6:
                                    self.curriculum_tfm.set_available_colours(9)
                                elif current_available_colours < 9:
                                    self.curriculum_tfm.set_available_colours(10)                              
                                elif current_available_colours < 27:
                                    self.curriculum_tfm.set_available_colours(27)
                                elif current_available_colours < 81:
                                    self.curriculum_tfm.set_available_colours(81)
                                elif current_available_colours < 243:
                                    self.curriculum_tfm.set_available_colours(243)
                                elif current_available_colours < 256:
                                    self.curriculum_tfm.set_available_colours(256)
  
                                best_loss = sys.float_info.max
                                best_epoch = epoch

                        if best_loss > result['val_loss']:
                            self.save_model(self.model, os.path.join(self.models_dir, self.model_name))
                    else:
                        return history

            self.end_epoch(epoch=epoch)
            if best_loss > result['val_loss']:   
                self.save_model(self.model, os.path.join(self.models_dir, self.model_name))
                best_loss = result['val_loss']
                best_epoch = epoch

        return history

    def train(self):
        logger.info('Training model...')

        optimizer = None
        train_scheduler = None
        momentum = None
        use_nesterov = None
        use_train_scheduler = self.config['optimizer']['use_train_scheduler'] if 'use_train_scheduler' in self.config['optimizer'] else False
        train_scheduler_config = None
        step_schedule_on_batch = False
        optimizer_config = self.config['optimizer']
        weight_decay =  self.config['weight_decay']
        if optimizer_config['name'] == 'sgd':
            learning_rate = optimizer_config['learning_rate']
            momentum = optimizer_config['momentum'] if 'momentum' in optimizer_config else 0.0
            use_nesterov = optimizer_config['use_nesterov'] if 'use_nesterov' in optimizer_config else False

            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=use_nesterov
            )

            if use_train_scheduler:
                train_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[60, 120, 160],
                    gamma=0.2
                )

                train_scheduler_config = {
                    "type": "MultiStepLR",
                    "train_scheduler_milestones": "[60, 120, 160]",
                    "train_scheduler_gamma": "0.2",
                }

                step_schedule_on_batch = False
            

        elif optimizer_config['name'] == 'adam':
            learning_rate = optimizer_config['learning_rate']
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                learning_rate,
                weight_decay=weight_decay
            )

            if use_train_scheduler:
                train_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    learning_rate,
                    epochs=self.config['epochs'], 
                    steps_per_epoch=len(self.training_dl)
                )
               
                train_scheduler_config = {
                    "type": "OneCycleLR",
                    "max_lr": learning_rate
                }
            
                step_schedule_on_batch = True

        else:
            raise ValueError('Unsupported optimizer {}'.format(self.config['optimizer']['name']))

        grad_clip = self.config['grad_clip'] if 'grad_clip' in self.config else None


        params = {
            "epochs": self.config['epochs'],
            "batch_size": self.config['batch_size'],
            "weight_decay": weight_decay,
            "optimizer": self.config['optimizer']['name'],
            "learning_rate": learning_rate,
            "momentum": momentum,
            "use_nesterov": use_nesterov,
            "train_scheduler_config": train_scheduler_config,
            "early_stop_enabled": self.config['early_stop_enabled'] if 'early_stop_enabled' in self.config else False,
            "early_stop_patience": self.config['early_stop_patience'] if 'early_stop_patience' in self.config else 10,
            "early_stop_min_delta":self.config['early_stop_min_delta'] if 'early_stop_min_delta' in self.config else 0.0001,
            "warmup_epochs":self.config['warmup_epochs'] if 'warmup_epochs' in self.config else 0,

        }

        logger.info('Using parameters: {}'.format(json.dumps(params)))
        if 'curriculum' in self.config:
            logger.info('Using curriculum  parameters: {}'.format(json.dumps(self.config['curriculum'])))

        torch.cuda.empty_cache()
        history = [eval_training(self.model, self.validation_dl)]
        start = t.time()
        history += self.fit(start, self.config['epochs'], self.model, optimizer, train_scheduler, step_schedule_on_batch, grad_clip)
        end = t.time()
        logger.info('Training model done (time taken: {}s)'.format(end - start))

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
