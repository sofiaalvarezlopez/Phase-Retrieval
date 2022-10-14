import sys
import logging

def set_logger(log_level, log_file, basic=False):
    """Sets up the logger configuration for the CNN training
    
    Keyword arguments:
    log_level: Level of logging set by the user
    log_file: If provided, file in which logging messages will be written to  
    format: If true, basic formatting is used (recommended for long log messages)
    """
    handlers = [logging.StreamHandler()]
    format = '' if basic else '%(asctime)s | %(name)s [%(levelname)s] %(message)s'
    if log_file: handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=log_level,
        format=format,
        force=True,
        handlers=handlers)


def disable_loggers(class_name):
    """Disables all loggers from imported modules
    
    Keyword arguments:
    class_name: Name of the class (which will be the logger name)
    """
    for log_name, log_obj in logging.Logger.manager.loggerDict.items():
     if log_name != class_name:
          log_obj.disabled = True

# Useful for logger
args = None
def set_args(new_args):
    global args
    args = new_args

def get_logger(name):
    '''Get logger according to commandline args'''
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()
    if hasattr(args, 'log_level'):
        log_level = getattr(logging, args.log_level)
    else:
        log_level = logging.INFO    
    logger.setLevel(log_level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    fmt = logging.Formatter('%(asctime)s | %(name)s [%(levelname)s] %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if hasattr(args, 'log_file') and args.log_file_name:
        fh = logging.FileHandler(args.log_file_name)
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


