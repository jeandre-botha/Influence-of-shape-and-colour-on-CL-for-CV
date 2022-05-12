import argparse
import logging
import json
import os

from jsonschema import validate
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from trainer import Trainer
from tester import Tester

LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "learning_rate": {"type": "number"}
    },
    "required": ["learning_rate"]
}

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='''Utility for training and testing a CNN''')
        parser.add_argument(
            '-a',
            '--action',
            type=str,
            nargs="?",
            required=True,
            choices=['train', 'test'],
            help='the action that should be performed for a model',
        )
        parser.add_argument(
            '-m',
            '--model',
            type=str,
            nargs="?",
            required=True,
            help='the path of the the model that should be used',
        )
        parser.add_argument(
            '-d',
            '--data',
            type=str,
            nargs="?",
            required=True,
            help='the path of training/testing',
        )
        parser.add_argument(
            '-c',
            '--config',
            type=str,
            nargs="?",
            required=False,
            help='the path to the config file. Required when training',
        )

        options = parser.parse_args()
        config = {}
        with open(options.config, 'r') as config_file:
            config = json.loads(config_file.read())
        
        try:
            validate(instance=config, schema=CONFIG_SCHEMA)
        except JsonSchemaValidationError as err:
            LOGGER.exception("Invalid config file structure, err: ", err)

        if options.action == "train":
            if options.config == None:
                print('Bad options provided.')
                parser.print_help()
            trainer = Trainer(os.path.abspath(options.model), os.path.abspath(options.data), config)
        elif options.action == "test":
            tester = Tester(os.path.abspath(options.model), os.path.abspath(options.data))

    except Exception as ex:
        LOGGER.exception(ex)