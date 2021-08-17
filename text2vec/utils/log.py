# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import logging

import colorlog

log_config = {
    'DEBUG': {
        'level': 10,
        'color': 'purple'
    },
    'INFO': {
        'level': 20,
        'color': 'green'
    },
    'TRAIN': {
        'level': 21,
        'color': 'cyan'
    },
    'EVAL': {
        'level': 22,
        'color': 'blue'
    },
    'WARNING': {
        'level': 30,
        'color': 'yellow'
    },
    'ERROR': {
        'level': 40,
        'color': 'red'
    },
    'CRITICAL': {
        'level': 50,
        'color': 'bold_red'
    }
}


def get_logger(name, log_file=None, log_level='DEBUG'):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :param log_level: 日志级别
    :return:
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(log_level.upper())
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s[%(levelname)7s %(asctime)s %(module)s:%(lineno)4d] %(message)s',
        log_colors={key: conf['color'] for key, conf in log_config.items()},
        datefmt='%Y%m%d %I:%M:%S')
    if log_file:
        f_handle = logging.FileHandler(log_file)
        f_handle.setFormatter(formatter)
        logger.addHandler(f_handle)
    handle = logging.StreamHandler()
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    return logger


logger = get_logger(__name__, log_file=None, log_level='DEBUG')


def set_log_level(log_level='INFO'):
    logger.setLevel(log_level.upper())


if __name__ == '__main__':
    logger.debug('hi-debug')
    logger.info('hi-info')
    logger.error('hi-error')
    logger.warning('hi-warn')
    set_log_level('info')
    logger.debug('hi2-debug')  # ignore
    logger.info('hi2-info')
    logger.error('hi2-error')
    logger.warning('hi2-warn')
