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


