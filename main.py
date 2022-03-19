import argparse
import logging.handlers
from configparser import ConfigParser

from hotdogconfig import Configuration
from hotdogcmdfactory import HotDogCmdFactory

LOG_FILE_NAME = "hotdog.log"
CFG_FILE_NAME = "config.ini"

logger = logging.getLogger("ACE")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger.setLevel(logging.DEBUG)

fileHandler = logging.handlers.RotatingFileHandler(LOG_FILE_NAME,
                                                   maxBytes=1000000,
                                                   backupCount=5
                                                   )
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
consoleHandler.setFormatter(formatter)

logger.addHandler(consoleHandler)

# cpltrain -cpl /home/jpvel/Workspace/hotdogworkspace/hotdog/data/LeadCampaign_Performance_final_dataset.csv -cpo /home/jpvel/Desktop/outcome2.csv
# cpltrain /home/jpvel/PycharmProjects/hotdog/data/LeadCampaign_Performance_final_dataset.csv
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Campaign  optimiization module')
    parser.add_argument('-c', '--command',
                        action="store",
                        dest="command",
                        help="Possible commands(prepup, feature, train, test, evaluate, ajfit1,ajtrain)",
                        required=False)
    parser.add_argument('-cpl', '--cpl-file',
                        action="store",
                        dest="cpl_file",
                        help="Provide an absolute path to the file that has cpl data",
                        required=False)
    parser.add_argument('-cpo', '--cpl-outcome',
                        action="store",
                        dest="cpl_out",
                        help="Provide an absolute path to the file that has cpl training outcome",
                        required=False)

    logger.info('ACE repl')
    inputConfig = parser.parse_args()
    config = ConfigParser()
    config.read([CFG_FILE_NAME], 'UTF-8')
    config_obj = Configuration(inputConfig, config)
    prompt = HotDogCmdFactory()
    prompt.config_obj = config_obj
    prompt.prompt = '>'
    prompt.cmdloop('Starting the  model training repl')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
