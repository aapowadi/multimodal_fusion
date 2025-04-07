import logging

def setup_logger(log_file_name, level=logging.INFO):
    """Function setup as many loggers as you want"""
    
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    file_handler = logging.FileHandler(log_file_name)   
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)

    return logger