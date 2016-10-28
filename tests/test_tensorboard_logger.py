# -*- coding: utf-8 -*-
import time

from tensorboard_logger import Logger, configure, log_value


def test_integration_default(tmpdir):
    configure(str(tmpdir), flush_secs=0.1)
    for step in range(10):
        log_value('v1', step * 1.5, step)
        log_value('v1', step ** 1.5 - 2, step)
    time.sleep(0.5)
    tf_log, = tmpdir.listdir()
    assert tf_log.basename.startswith('events.out.tfevents.')


def test_integration_logger(tmpdir):
    logger = Logger(str(tmpdir), flush_secs=0.1)
    for step in range(10):
        logger.log_value('v1', step * 1.5, step)
        logger.log_value('v1', step ** 1.5 - 2, step)
    time.sleep(0.5)
    tf_log, = tmpdir.listdir()
    assert tf_log.basename.startswith('events.out.tfevents.')
