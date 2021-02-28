import logging
import logging.handlers
import time
import datetime
import os


def get_log():
    filename_as_time = time.strftime("%m%d-%H%M%S")

    if not os.path.exists('log'):
        os.makedirs('log', exist_ok=True)

    logger = logging.getLogger("crumbs")
    logger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler(f"log/{filename_as_time}.txt")
    streamHandler = logging.StreamHandler()

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger


def train_summary(args, logger):

    logger.info("\n")
    logger.info("="*50)
    logger.info(f"Time : {datetime.datetime.now()}")
    logger.info("-"*30)
    logger.info(f"Learning Rate : {args['LEARNING_RATE']}")
    logger.info(f"Weight Decay : {args['WEIGHT_DECAY']}")
    logger.info(f"Epochs per Fold : {args['NUM_EPOCHS']}")
    logger.info(f"Batch Size : {args['BATCH_SIZE']}")
    logger.info("="*50+'\n')
