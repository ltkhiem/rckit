import json

class RCKit:
    def __init__(config_path=None):
        self.config = config

    def __ensure_config(func):
        def check(self):
            assert self.config is not None, "Config is not set"
            self.func()

    def load_config(config_path):
        self.config = json.load(open(config_path, 'r'))
    

            
        
        
