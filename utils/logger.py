import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir: str, level=logging.INFO):
        logger_name = f'trainer_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        os.makedirs(log_dir, exist_ok=True)
        
        log_filename = f'train_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
        log_filepath = os.path.join(log_dir, log_filename)
        
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
        self.logger.propagate = False
    
    def get_logger(self):
        return self.logger
    
    def log_hyperparams(self, params: dict):
        self.logger.info("--------- Hyperparameters ---------")
        for key, value in params.items():
            self.logger.info(f"{key:<15}: {value}")
        self.logger.info("-----------------------------------")