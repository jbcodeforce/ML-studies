import os, yaml

CONFIG_FILE= "config/config.yaml"


def load_config():
    config={}
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
    return config