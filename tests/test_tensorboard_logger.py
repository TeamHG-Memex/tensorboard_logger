# -*- coding: utf-8 -*-
import time

from tensorboard_logger import Logger, configure, log_value


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
