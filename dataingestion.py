import pandas as pd
import logging

logger = logging.getLogger("ACE")


class DataIngestion:

    def __init__(self, file_name):
        self.file_name = file_name

    def prepUp(self):
        try:
            data = pd.read_csv(self.file_name)
            logger.info(f'column listing #{data.columns}')
        except(RuntimeError) as error:
            logger.error("Error preparing data ", error.original_traceback)
            pass
