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
        'max_lr': {'type': 'number'},
        'grad_clip': {'type': 'number'},
        'optimizer': {'type': 'string'},
        'root_path': {'type': 'string'},
        'curriculum': {'type': 'object'},

    },
    'required': ['batch_size', 'epochs', 'weight_decay', 'max_lr', 'grad_clip', 'optimizer', 'root_path']
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