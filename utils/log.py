import logging
import sys
import os


class MyNetLogger:
    
    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        self.log_level = logging.DEBUG
        self.log_file_path = None
        self.log_format = '%(asctime)s-%(levelname)s : %(message)s'

    def set_log_level(self, level):
        self.log_level = level

    def set_log_file_path(self, file_path):
        self.log_file_path = file_path

    def set_log_format(self, log_format):
        self.log_format = log_format

    def init(self):
        logging.basicConfig(
            level=logging.DEBUG,
            filename='tmp.log'
        )
        formatter = logging.Formatter(self.log_format)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(self.log_level)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        if self.log_file_path is not None:
            fh = logging.FileHandler(self.log_file_path, encoding='utf-8')
            fh.setLevel(self.log_level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def __call__(self, msg):
        self.logger.info(msg)

    @staticmethod
    def default(log_path):
        if not os.path.isdir(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        log = MyNetLogger(None)
        log.set_log_file_path(log_path)
        log.init()
        return log

