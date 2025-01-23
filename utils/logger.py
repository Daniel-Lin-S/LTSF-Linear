import logging
from abc import ABC, abstractmethod
import traceback


# TODO - add an error logger to record failed experiments

class Logger(ABC):
    @abstractmethod
    def log(self, message: str, level: str):
        """
        Log a message.

        Parameters
        ----------
        message : str
            The message to log.
        level : str, optional
            Logging level. (e.g. info, debug)
        """
        pass


class DualLogger(Logger):
    """
    Log messages into both console and log file.
    A separate file errors.log will be created
    to store all error messages to
    track which experiments failed.
    """
    def __init__(self, log_file: str):
        """
        Initialise the Logger.

        Parameters
        ----------
        log_file : str
            Path to the log file where all detailed messages
            will be stored.
        """
        self.logger = logging.getLogger("TrainingLogger")
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        error_file_handler = logging.FileHandler('errors.log')
        error_file_handler.setLevel(logging.ERROR)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_formatter = logging.Formatter(
            "%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_file_handler)

    def log(self, message: str, level: str = "info",
            console_only: bool=False):
        """
        Log a message.

        Parameters
        ----------
        message : str
            The message to log.
        level : str, optional
            Logging level: 'info', 'debug', 'warning',
            'error', or 'critical'.
            Default is 'info'.
            If 'debug', the message will NOT be displayed
            in console, but can be found in the log file.
            All other levels will be logged to both
            console and log file.
        console_only : bool, optional
            if true, the message will only be logged to console
            but not the log file.
        """
        if console_only:
            # Temporarily disable the file handler
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)

        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        else:
            raise ValueError(
                'logging level can only be '
                'one of [info, debug, warning, error, critical]'
            )

        if console_only:
            # Re-add the file handler
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    self.logger.addHandler(handler)

    def stage_start(self, stage_name: str, setting: str):
        """
        Log the start of a stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage (e.g., "Training", "Validation").
        """
        self.log(f"Starting {stage_name} stage >>>>>>>>>>>>>>>>>>>>>>>",
                 level="info")
        self.log(f'Settings: {setting}', level='debug')

    def stage_end(self, stage_name: str):
        """
        Log the finish of a stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage (e.g., "Training", "Validation").
        """
        self.log(f">>>>>>>>>>>>>>>>>>>>>>> Finished {stage_name} stage.",
                 level="info")
    
    def stage_failed(self, error : Exception, stage_name: str):
        """
        Log the finish of a stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage (e.g., "Training", "Validation").
        """
        self.log_error(error, stage_name)
        self.log(f">>>>>>>>>>>>>>>>>>>>>>> {stage_name} failed, please check errors.log",
                 level="info")
        
    def start_experiment(self, exp_name: str, status: str):
        self.log('=============================================='
                 f' Experiment {exp_name}:{status} '
                 '==============================================',
                 level='info')
        
    def log_error(self, error : Exception, stage_name : str):
        """Log detailed error message and tracebacks"""
        self.log(f"{stage_name}: {type(error).__name__} occurred. ", 'error')
        self.log(traceback.format_exc(), 'error')
