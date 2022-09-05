import logging

# create logger
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)