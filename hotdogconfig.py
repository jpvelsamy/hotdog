import os
from os.path import expanduser


class Configuration:

    def __init__(self, commandLineArgument, sysCfg):
        self.commandName = commandLineArgument.command
        self.sourcefolder = sysCfg['prepsource']['sourcefolder']


    def get_source_folder(self):
        return self.sourcefolder
