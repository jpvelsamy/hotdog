import logging
from cmd import Cmd

from bostonregression import BostonRegression
from preparedata import PrepareData

logger = logging.getLogger("ACE")


class HotDogPrompt(Cmd):
    commands = ['prepup', 'feature', 'train', 'test', 'eval', 'bostonmodel', 'calimeta', 'calihead', 'calihead','bostonmeta','kfold']
    configObj = None

    def do_prepup(self, args):
        try:
            prepareData = PrepareData(self.configObj)
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
