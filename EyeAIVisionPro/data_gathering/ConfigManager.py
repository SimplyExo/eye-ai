import configparser
import os
from pathlib import Path

class ConfigManager:
    config = configparser.ConfigParser()

    def __init__(self):
        if not os.path.exists("config.conf"):
            self.generate_config()
        else:
            self.config.read("config.conf")

    def generate_config(self):
        self.config['SETTINGS'] = {
            'OutputDir': '/mnt/output',
            'CaptureDelay': 10,
            'ImageWidth': 1280,
            'ImageHeight': 720
        }

        with open("config.conf", "w") as f:
            self.config.write(f)

    def get_outputdir(self):
        return Path(self.config['SETTINGS']["OutputDir"])

    def get_capturedelay(self):
        return int(self.config['SETTINGS']["CaptureDelay"])

    def get_width(self):
        return int(self.config['SETTINGS']["ImageWidth"])

    def get_height(self):
        return int(self.config['SETTINGS']["ImageHeight"])
    