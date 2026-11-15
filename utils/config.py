import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_config(path="config.json"):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)

def resolve(path):
    # paths in config are relative to the project root
    return os.path.join(ROOT, path)

cfg = load_config()
