import logging
from cmd import Cmd

from bostonregression import BostonRegression
from preparedata import PrepareData
from testaskjunoace import TestAskJunoACE
from askjunoace import  AskJunoACE
logger = logging.getLogger("ACE")


class HotDogCmdFactory(Cmd):
    commands = ['prepup', 'feature', 'train', 'test', 'eval', 'bostonmodel', 'calimeta', 'calihead', 'calihead','bostonmeta','kfold','ajfit','ajfit2','cpltrain']
    config_obj = None


    def do_cpltrain(self, input_file):
        try:
            logger.info(f'Initialising ace engine')
            aj = AskJunoACE(self.config_obj, input_file)
            logger.info(f'Loading data for the ace engine to get ready')
            aj.load_data()
            logger.info(f'Normalising data for the ace engine to get ready')
            aj.normalize_data()
            logger.info(f'Splitting training and testing data for the ace engine ')
            aj.test_train_split()
            logger.info(f'Initializing the deep learning engine')
            aj.model_init()
            logger.info(f'Training the deep learning engine')
            aj.model_train()
        except(RuntimeError, TypeError, NameError) as error:
            logger.error("Error preparing data ", error.original_traceback)
            pass

    def do_ajfit(self, args):
        try:
            aj = TestAskJunoACE()
            aj.fit_1('/home/jpvel/Workspace/hotdogworkspace/hotdog/data/LeadCampaign_Performance_final_dataset.csv')
        except(RuntimeError, TypeError, NameError) as error:
            logger.error("Error preparing data ", error.original_traceback)
            pass

    def do_ajfit2(self, args):
        try:
            aj = TestAskJunoACE()
            aj.fit_2('/home/jpvel/Workspace/hotdogworkspace/hotdog/data/LeadCampaign_Performance_final_dataset.csv')
        except(RuntimeError, TypeError, NameError) as error:
            logger.error("Error preparing data ", error.original_traceback)
            pass

    def do_prepup(self, args):
        try:
            prepareData = PrepareData(self.config_obj)
            prepareData.execute()
        except(RuntimeError, TypeError, NameError) as error:
            logger.error("Error preparing data ", error.original_traceback)
            pass

    def do_bostonmeta(self, args):
        br = BostonRegression()
        br.showbostonmeta()

    def do_calihead(self, args):
        br = BostonRegression()
        br.showhead()

    def do_calimeta(self, args):
        br = BostonRegression()
        br.showmetadata()

    def do_bostonmodel(self, args):
        br = BostonRegression()
        br.build_model()
        br.train_model()
        br.test()

    def do_kfold(self, args):
        br = BostonRegression()
        br.kfold_validation()
        br.plot_kfold()

    def do_quit(self, args):
        """Quits the program."""
        logger.info("Quitting")
        raise SystemExit
