class Configuration:

    def __init__(self, cmdlinearg, sysCfg):
        self.commandName = cmdlinearg.command
        self.sourcefolder = sysCfg['prepsource']['sourcefolder']
        self.cpltrainfile = cmdlinearg.cpl_file
        self.cpl_names = sysCfg['cpl']['names']
        self.cploutcome = cmdlinearg.cpl_out

    def get_source_folder(self):
        return self.sourcefolder

    def get_cpl_train_file(self):
        return self.cpltrainfile

    def get_cpl_features(self):
        return self.cpl_names

    def get_cpl_outcome(self):
        return self.cploutcome

