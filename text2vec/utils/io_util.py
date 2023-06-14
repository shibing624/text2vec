# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
import os


def load_jsonl(json_path):
    """
    Load jsonl file.
    Args:
        json_path (str): jsonl file path.
    Returns:
        list: list of json object.
    """
    data = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for json_str in f:
            try:
                result = json.loads(json_str.strip('\n'))
                data.append(result)
            except:
                print('error', json_str)
    return data


def save_jsonl(data_list, json_path, mode='w', encoding='utf-8'):
    dir = os.path.dirname(os.path.abspath(json_path))
    if not os.path.exists(dir):
        print(dir)
        os.makedirs(dir)
    with open(json_path, mode=mode, encoding=encoding) as f:
        for entry in data_list:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
    print(f'save to {json_path}, size: {len(data_list)}')


def load_json(json_path, encoding='utf-8'):
    """
    Load json file.
    Args:
        json_path: json file path.
        encoding: json file encoding.
    Returns:
        data: list of dict.
    """
    with open(json_path, mode='r', encoding=encoding) as f:
        data = json.load(f)
    return data


def save_json(data, json_path, mode='w', encoding='utf-8'):
    dir = os.path.dirname(os.path.abspath(json_path))
    if not os.path.exists(dir):
        print(dir)
        os.makedirs(dir)
    with open(json_path, mode=mode, encoding=encoding) as f:
        f.write(json.dumps(data, ensure_ascii=False))
