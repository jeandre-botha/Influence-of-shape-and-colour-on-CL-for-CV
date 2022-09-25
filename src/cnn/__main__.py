import argparse
import json
import os
from os.path import exists as file_exists

from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from logger import logger
from trainer import Trainer

import warnings
warnings.filterwarnings('ignore')

CONFIG_SCHEMA = {
    'type': 'object',
    'properties': {
        'batch_size': {'type': 'number'},
        'epochs': {'type': 'number'},
        'weight_decay': {'type': 'number'},
        'grad_clip': {'type': 'number'},
        'num_classes': {'type': 'number'},
        'early_stop_enabled': {'type': 'boolean'},
        'early_stop_patience': {'type': 'number'},
        'early_stop_min_delta': {'type': 'number'},
        'optimizer': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'learning_rate':  {'type': 'number'},
                'momentum': {'type': 'number'},
                'use_nesterov': {'type': 'boolean'},
                'use_train_scheduler': {'type': 'boolean'},
            },
            'required': ['name', 'learning_rate']
        },
        'root_path': {'type': 'string'},
        'curriculum': {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'}
            },
            'required': ['name']
        },

    },
    'required': ['batch_size', 'epochs', 'weight_decay', 'optimizer', 'root_path', 'num_classes']
}


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='''Utility for training and testing a CNN''')
        parser.add_argument(
            '-a',
            '--action',
            type=str,
            nargs='?',
            required=True,
            choices=['train', 'test', 'both'],
            help='the action that should be performed for a model',
        )
        parser.add_argument(
            '-m',
            '--model',
            type=str,
            nargs='?',
            required=True,
            help='the name of the the model',
        )
        parser.add_argument(
            '-d',
            '--dataset',
            type=str,
            nargs='?',
            required=True,
            help='the name of the keras datasetr',
        )
        parser.add_argument(
            '-c',
            '--config',
            type=str,
            nargs='?',
            required=False,
            help='the path to the config file. Required when training',
        )

        options = parser.parse_args()

        if options.config == None:
            logger.error('Bad options provided.')
            parser.print_help()
            exit(0)

        if not file_exists(options.config):
            logger.error('config file does not exist at location "'+ options.config +'"')
            exit(0)

        config = {}
        with open(options.config, 'r') as config_file:
            config = json.loads(config_file.read())
        
        try:
            validate(instance=config, schema=CONFIG_SCHEMA)
        except JsonSchemaValidationError as err:
            logger.error('Invalid config file structure, err: ', err)
            exit(0)

        trainer = Trainer(options.model, options.dataset, config)

        if options.action == 'train':
            trainer.train()
        elif options.action == 'test':
            trainer.test()
        elif options.action == 'both':
            trainer.train()
            trainer.test()

    except Exception as ex:
        logger.exception(ex)