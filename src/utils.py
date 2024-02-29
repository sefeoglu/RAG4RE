import os
import json
import logging
import sys

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
