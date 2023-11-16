import logging
from utils.settings import DISPLAY_DEBUG_MESSAGES

# Define the basic logger to display the messages on the console
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG if DISPLAY_DEBUG_MESSAGES else logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
