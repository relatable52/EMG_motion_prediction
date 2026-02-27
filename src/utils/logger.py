import logging
import os
import datetime

from dotenv import load_dotenv

load_dotenv()

def setup_logger(log_file: str) -> logging.Logger:
    """
    Sets up a logger that prints messages to the console and saves them to a file.
    Args:
        log_file (str): The path to the log file.
        Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger('gatech_data_analysis')
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

# Create a unique log file name based on the current date and time
log_file_name = f"project_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
log_file_path = os.path.join(os.getenv('LOG_DIR'), log_file_name)

# Set up a logger instance for the project and specify the log file path
logger = setup_logger(log_file_path)