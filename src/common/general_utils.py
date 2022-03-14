import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def print_args(params):
    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in vars(params).items():
        key_str = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s =   %s", key_str, val)
