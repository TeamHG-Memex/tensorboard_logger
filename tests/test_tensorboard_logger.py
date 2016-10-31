# -*- coding: utf-8 -*-
import time

from tensorboard_logger import Logger, configure, log_value
from tensorboard_logger.tensorboard_logger import make_valid_tf_name


def test_integration_default(tmpdir):
    configure(str(tmpdir), flush_secs=0.1)
    for step in range(10):
        log_value('v1', step * 1.5, step)
        log_value('v2', step ** 1.5 - 2, step)
    time.sleep(0.5)
    tf_log, = tmpdir.listdir()
    assert tf_log.basename.startswith('events.out.tfevents.')


def test_integration_logger(tmpdir):
    logger = Logger(str(tmpdir), flush_secs=0.1)
    for step in range(10):
        logger.log_value('v1', step * 1.5, step)
        logger.log_value('v2', step ** 1.5 - 2, step)
    time.sleep(0.5)
    tf_log, = tmpdir.listdir()
    assert tf_log.basename.startswith('events.out.tfevents.')


def test_dummy():
    logger = Logger(None, is_dummy=True)
    for step in range(3):
        logger.log_value('A v/1', step, step)
        logger.log_value('A v/2', step * 2, step)
    assert dict(logger.dummy_log) == {
        'A_v/1': [(0, 0), (1, 1), (2, 2)],
        'A_v/2': [(0, 0), (1, 2), (2, 4)],
    }


def test_make_valid_tf_name():
    mvn = make_valid_tf_name
    assert mvn('This/is/valid') == 'This/is/valid'
    assert mvn('0-This/is/valid') == '0-This/is/valid'
    assert mvn('.This/is/valid') == '.This/is/valid'
    assert mvn(' This/is invalid') == '._This/is_invalid'
    assert mvn('-This-is-invalid') == '.-This-is-invalid'


def test_unique():
    logger = Logger(None, is_dummy=True)
    for step in range(1, 3):
        # names that normalize to the same valid name
        logger.log_value('A v/1', step, step)
        logger.log_value('A\tv/1', step * 2, step)
        logger.log_value('A  v/1', step * 3, step)
    assert dict(logger.dummy_log) == {
        'A_v/1':   [(1, 1), (2, 2)],
        'A_v/1/1': [(1, 2), (2, 4)],
        'A_v/1/2': [(1, 3), (2, 6)],
    }
