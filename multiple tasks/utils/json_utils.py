import datetime
import json
from types import GeneratorType as Generator
from collections.abc import Iterable
from ast import literal_eval
from torch import Tensor
import numpy as np


def default_json_handler(obj):
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, Tensor):
        return obj.tolist()
    for np_int_type in (np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64):
        if isinstance(obj, np_int_type):
            return int(obj)
    for np_float_type in (np.float32, np.float64):
        if isinstance(obj, np_float_type):
            return float(obj)
    if isinstance(obj, (Generator, Iterable)):
        return tuple(obj)
    raise Exception(f'Cannot dump data type: {type(obj)}')


def save(obj, path, indent=4):
    print(f'Saving {type(obj)} object to {path}...')
    with open(path, 'w+') as f:
        json.dump(obj, f, indent=indent, default=default_json_handler)
    print(f'Saved {type(obj)} object to {path}')


def save_all(objs, paths, indent=4):
    for i in range(len(objs)):
        save(objs[i], paths[i], indent=indent)


def load(path):
    print(f'Loading {path}...')
    with open(path) as f:
        obj = json.load(f)
        print(f'Loaded {path} to {type(obj)} object')
        return obj


def load_all(paths):
    return [load(path) for path in paths]


def jsonify(obj):
    try:
        return json.loads(obj)
    except:
        return obj


def key2str(dictionary, original_types=(tuple,)):
    for key in list(dictionary.keys()):
        if isinstance(dictionary[key], dict):
            key2str(dictionary[key], original_types)
        if isinstance(key, original_types):
            dictionary[str(key)] = dictionary[key]
            del dictionary[key]


def str2key(dictionary):
    for key in list(dictionary.keys()):
        if isinstance(dictionary[key], dict):
            str2key(dictionary[key])
        if isinstance(key, str):
            try:
                dictionary[literal_eval(key)] = dictionary[key]
            except:
                continue
            del dictionary[key]
