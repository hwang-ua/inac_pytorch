import os
import sys

import logging

# from tensorboardX import SummaryWriter


class Logger:
    def __init__(self, config):
        log_file = os.path.join(config.get_log_dir(), 'log')
        self._logger = logging.getLogger()

        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s | %(message)s')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

        self._logger.setLevel(level=logging.INFO)

        self.config = config
        # if config.tensorboard_logs: self.tensorboard_writer = SummaryWriter(config.get_log_dir())

    def info(self, log_msg):
        self._logger.info(log_msg)