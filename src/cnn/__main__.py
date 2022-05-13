import argparse
import json
import os
from os.path import exists as file_exists

from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from logger import logger
from trainer import Trainer
from tester import Tester

import warnings
warnings.filterwarnings('ignore')

CONFIG_SCHEMA = {
    'type': 'object',
    'properties': {
        'batch_size': {'type': 'number'},
        'epochs': {'type': 'number'}
    },
    'required': ['batch_size', 'epochs']
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
            choices=['train', 'test'],
            help='the action that should be performed for a model',
        )
        parser.add_argument(
            '-m',
            '--model',
            type=str,
            nargs='?',
            required=True,
            help='the path of the the model that should be used',
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

        if options.action == 'train' and options.config == None:
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

        if options.action == 'train':
            trainer = Trainer(options.model, options.dataset, config)
            trainer.train()
        elif options.action == 'test':
            tester = Tester(options.model, options.dataset, config)
            tester.test()

    except Exception as ex:
        logger.exception(ex)