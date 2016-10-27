# -*- coding: utf-8 -*-
import os
from contextlib import contextmanager
import tempfile
import shutil
import time

from tensorboard_logger import Logger, configure, log_value


@contextmanager
def make_tempdir():
    tempdir = tempfile.mkdtemp()
    try:
        yield tempdir
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def test_integration_default():
    with make_tempdir() as tempdir:
        run_dir = os.path.join(tempdir, 'run-1')
        configure(run_dir, flush_secs=0.1)
        for step in range(10):
            log_value('v1', step * 1.5, step)
            log_value('v1', step ** 1.5 - 2, step)
        time.sleep(0.5)
        tf_log, = os.listdir(run_dir)
        assert tf_log.startswith('events.out.tfevents.')


def test_integration_logger():
    with make_tempdir() as tempdir:
        run_dir = os.path.join(tempdir, 'run-1')
        logger = Logger(run_dir, flush_secs=0.1)
        for step in range(10):
            logger.log_value('v1', step * 1.5, step)
            logger.log_value('v1', step ** 1.5 - 2, step)
        time.sleep(0.5)
        tf_log, = os.listdir(run_dir)
        assert tf_log.startswith('events.out.tfevents.')
