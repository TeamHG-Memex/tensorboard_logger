# -*- coding: utf-8 -*-
import time
import os
import glob
import numpy as np

from tensorboard_logger import Logger, configure, log_value
from tensorboard_logger.tensorboard_logger import make_valid_tf_name


def test_smoke_default(tmpdir):
    configure(str(tmpdir), flush_secs=0.1)
    for step in range(10):
        log_value('v1', step * 1.5, step)
        log_value('v2', step ** 1.5 - 2)
    time.sleep(0.5)
    tf_log, = tmpdir.listdir()
    assert tf_log.basename.startswith('events.out.tfevents.')


def test_smoke_logger(tmpdir):
    logger = Logger(str(tmpdir), flush_secs=0.1)
    for step in range(10):
        logger.log_value('v1', step * 1.5, step)
        logger.log_value('v2', step ** 1.5 - 2)
    time.sleep(0.5)
    tf_log, = tmpdir.listdir()
    assert tf_log.basename.startswith('events.out.tfevents.')


def test_serialization(tmpdir):
    logger = Logger(str(tmpdir), flush_secs=0.1, dummy_time=256.5)
    logger.log_value('v/1', 1.5, 1)
    logger.log_value('v/22', 16.0, 2)
    time.sleep(0.5)
    tf_log, = tmpdir.listdir()
    assert tf_log.read_binary() == (
        # step = 0, initial record
        b'\x18\x00\x00\x00\x00\x00\x00\x00\xa3\x7fK"\t\x00\x00\x00\x00\x00\x08p@\x1a\rbrain.Event:2\xbc\x98!+'
        # v/1
        b'\x19\x00\x00\x00\x00\x00\x00\x00\x8b\xf1\x08(\t\x00\x00\x00\x00\x00\x08p@\x10\x01*\x0c\n\n\n\x03v/1\x15\x00\x00\xc0?,\xec\xc0\x87'
        # v/22
        b'\x1a\x00\x00\x00\x00\x00\x00\x00\x12\x9b\xd8-\t\x00\x00\x00\x00\x00\x08p@\x10\x02*\r\n\x0b\n\x04v/22\x15\x00\x00\x80A\x8f\xa3\xb6\x88'
    )


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


def test_dummy_histo():
    logger = Logger(None, is_dummy=True)
    bins = [0, 1, 2, 3]
    logger.log_histogram('key', (bins, [0.0, 1.0, 2.0]), step=1)
    logger.log_histogram('key', (bins, [1.0, 1.5, 2.5]), step=2)
    logger.log_histogram('key', (bins, [0.0, 1.0, 2.0]), step=3)

    assert dict(logger.dummy_log) == {
        'key': [(1, (bins, [0.0, 1.0, 2.0])),
                (2, (bins, [1.0, 1.5, 2.5])),
                (3, (bins, [0.0, 1.0, 2.0]))]}


def test_real_histo_tuple(tmpdir):
    """
    from tests.test_tensorboard_logger import *
    import ubelt as ub
    ub.delete(ub.ensure_app_cache_dir('tf_logger'))
    tmpdir = ub.ensure_app_cache_dir('tf_logger/runs/run1')
    """
    logger = Logger(str(tmpdir), flush_secs=0.1)
    bins = [-.5, .5, 1.5, 2.5]
    logger.log_histogram('hist1', (bins, [0.0, 1.0, 2.0]), step=1)
    logger.log_histogram('hist1', (bins, [1.0, 1.5, 2.5]), step=2)
    logger.log_histogram('hist1', (bins, [0.0, 1.0, 2.0]), step=3)
    tf_log, = glob.glob(str(tmpdir) + '/*')
    assert os.path.basename(tf_log).startswith('events.out.tfevents.')


def test_real_histo_data(tmpdir):
    logger = Logger(str(tmpdir), flush_secs=0.1)
    logger.log_histogram('hist2', [1, 7, 6, 9, 8, 1, 4, 5, 3, 7], step=1)
    logger.log_histogram('hist2', [5, 3, 2, 0, 8, 5, 7, 7, 7, 2], step=2)
    logger.log_histogram('hist2', [1, 2, 2, 1, 5, 1, 8, 4, 4, 1], step=3)
    tf_log, = glob.glob(str(tmpdir) + '/*')
    assert os.path.basename(tf_log).startswith('events.out.tfevents.')


def test_dummy_images():
    logger = Logger(None, is_dummy=True)
    img = np.random.rand(10, 10)
    images = [img, img]
    logger.log_images('key', images, step=1)
    logger.log_images('key', images, step=2)
    logger.log_images('key', images, step=3)

    assert dict(logger.dummy_log) == {
        'key': [(1, images),
                (2, images),
                (3, images)]}


def test_real_image_data(tmpdir):
    logger = Logger(str(tmpdir), flush_secs=0.1)
    img = np.random.rand(10, 10)
    images = [img, img]
    logger.log_images('key', images, step=1)
    logger.log_images('key', images, step=2)
    logger.log_images('key', images, step=3)
    tf_log, = glob.glob(str(tmpdir) + '/*')
    assert os.path.basename(tf_log).startswith('events.out.tfevents.')
