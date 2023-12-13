import json

cached_config = None

def get_config():
    global cached_config
    if cached_config is None:
        with open('config.json') as json_file:
            cached_config = json.load(json_file)
    return cached_config
