import os
import sys
import yaml


class ConfigUtil:
    def __init__(self, path):
        self.path = path
        return

    def get_config(self, config):
        conf = open(os.path.join(self.path, config)).read()
        try:
            return self.parse_config(conf)
        except Exception as e:
            print(e)

    def parse_config(self, config):
        config = yaml.safe_load(config)
        config = self._walk(config)
        return config

    def _walk(self, config):
        if not isinstance(config, str):
            for field, value in config.items():
                if not isinstance(value, str):
                    self._walk(value)
                else:
                    val = value.replace("${ROOT_DIR}", self.path)
                    config[field] = val
        return config