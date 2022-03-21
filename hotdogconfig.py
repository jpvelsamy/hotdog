class Configuration:

    def __init__(self, cmdlinearg, sys_cfg):
        self.path_separator = sys_cfg['common']['path_seperator']
        self.commandName = cmdlinearg.command
        self.sourcefolder = sys_cfg['prepsource']['sourcefolder']
        self.cpltrainfile = cmdlinearg.cpl_file
        self.cpl_names = sys_cfg['cpl']['names']
        self.cploutcome = sys_cfg['cpl']['outcomefolder']
        self.cpl_ratio = sys_cfg['cpl']['ratio']
        self.model_save_path = sys_cfg['cpl']['modelsavepath']
        self.input_file = sys_cfg['cpl']['input_file']
        self.test_file = sys_cfg['cpl']['test_file']
        self.test_features = sys_cfg['cpl']['test_names']


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
        return self.model_save_path

    def get_input_file_name(self):
        return self.input_file

    def get_test_file_name(self):
        return self.test_file

    def get_path_separator(self):
        return self.path_separator

    def get_test_feature_names(self):
        return self.test_features