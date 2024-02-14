import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
                    handlers=[
                        RotatingFileHandler(LOG_FILE_PATH, maxBytes=1024 * 1024, backupCount=5),  # Rotating file handler
                        logging.StreamHandler()  # Console handler
                    ])

# Get the logger
logger = logging.getLogger(__name__)

