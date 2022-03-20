class Configuration:

    def __init__(self, cmdlinearg, sys_cfg):
        self.commandName = cmdlinearg.command
        self.sourcefolder = sys_cfg['prepsource']['sourcefolder']
        self.cpltrainfile = cmdlinearg.cpl_file
        self.cpl_names = sys_cfg['cpl']['names']
        self.cploutcome = sys_cfg['cpl']['outcomefolder']
        self.cpl_ratio = sys_cfg['cpl']['ratio']
        self.model_save_path = sys_cfg['cpl']['modelsavepath']

    def get_source_folder(self):
        return self.sourcefolder

    def get_cpl_train_file(self):
        return self.cpltrainfile

    def get_cpl_features(self):
        return self.cpl_names

    def get_cpl_outcome(self):
        return self.cploutcome

    def get_cpl_ratio(self):
        return self.cpl_ratio

    def get_model_save_path(self):
        return self.model_save_path;