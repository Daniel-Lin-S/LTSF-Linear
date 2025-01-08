import logging
from abc import ABC, abstractmethod


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

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_formatter = logging.Formatter(
            "%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, message: str, level: str = "info"):
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
        """
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

    def start_stage(self, stage_name: str, setting: str):
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

    def finish_stage(self, stage_name: str):
        """
        Log the finish of a stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage (e.g., "Training", "Validation").
        """
        self.log(f">>>>>>>>>>>>>>>>>>>>>>> Finished {stage_name} stage.",
                 level="info")
        
    def start_experiment(self, exp_name: str, status: str):
        self.log('=============================================='
                 f' Experiment {exp_name}:{status} '
                 '==============================================',
                 level='info')
