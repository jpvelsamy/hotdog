import logging

logger = logging.getLogger("ACE")

class PrepareData:

    def __init__(self, configObject):
        self.configObject = configObject
        self.srcFolder = self.configObject.get_source_folder()

    def execute(self):
        logger.info("Retrieving data from source folder %s", self.srcFolder)
