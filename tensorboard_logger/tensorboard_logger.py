# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import re
import socket
import struct
import time
import numpy as np

import six

import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

try:
    from tensorflow.core.util import event_pb2
    from tensorflow.core.framework import summary_pb2
except ImportError:
    from .tf_protobuf import summary_pb2, event_pb2
from .crc32c import crc32c


__all__ = ['Logger', 'configure', 'unconfigure', 'log_value', 'log_histogram', 'log_images']


_VALID_OP_NAME_START = re.compile('^[A-Za-z0-9.]')
_VALID_OP_NAME_PART = re.compile('[A-Za-z0-9_.\\-/]+')


class Logger(object):
    def __init__(self, logdir, flush_secs=2, is_dummy=False, dummy_time=None):
        self._name_to_tf_name = {}
        self._tf_names = set()
        self.is_dummy = is_dummy
        self.logdir = logdir
        self.flush_secs = flush_secs  # TODO
        self._writer = None
        self._dummy_time = dummy_time
        if is_dummy:
            self.dummy_log = defaultdict(list)
        else:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            hostname = socket.gethostname()
            filename = os.path.join(
                self.logdir, 'events.out.tfevents.{}.{}'.format(
                    int(self._time()), hostname))
            self._writer = open(filename, 'wb')
            self._write_event(event_pb2.Event(
                wall_time=self._time(), step=0, file_version='brain.Event:2'))

    def _ensure_tf_name(self, name):
        if not isinstance(name, six.string_types):
            raise TypeError('"name" should be a string, got {}'
                            .format(type(name)))
        try:
            tf_name = self._name_to_tf_name[name]
        except KeyError:
            tf_name = self._make_tf_name(name)
            self._name_to_tf_name[name] = tf_name
        return tf_name

    def _check_step(self, step):
        if step is not None and not isinstance(step, six.integer_types):
            raise TypeError('"step" should be an integer, got {}'
                            .format(type(step)))

    def log_value(self, name, value, step=None):
        """Log new value for given name on given step.

        Args:
            name (str): name of the variable (it will be converted to a valid
                tensorflow summary name).
            value (float): this is a real number to be logged as a scalar.
            step (int): non-negative integer used for visualization: you can
                log several different variables on one step, but should not log
                different values of the same variable on the same step (this is
                not checked).
        """
        if isinstance(value, six.string_types):
            raise TypeError('"value" should be a number, got {}'
                            .format(type(value)))
        value = float(value)

        self._check_step(step)
        tf_name = self._ensure_tf_name(name)

        summary = self._scalar_summary(tf_name, value, step)
        self._log_summary(tf_name, summary, value, step=step)

    def log_histogram(self, name, value, step=None):
        """Log a histogram for given name on given step.

        Args:
            name (str): name of the variable (it will be converted to a valid
                tensorflow summary name).
            value (tuple or list): either list of numbers
                to be summarized as a histogram, or a tuple of bin_edges and
                bincounts that directly define a histogram.
            step (int): non-negative integer used for visualization
        """
        if isinstance(value, six.string_types):
            raise TypeError('"value" should be a number, got {}'
                            .format(type(value)))

        self._check_step(step)
        tf_name = self._ensure_tf_name(name)

        summary = self._histogram_summary(tf_name, value, step=step)
        self._log_summary(tf_name, summary, value, step=step)

    def log_images(self, name, images, step=None):
        """Log new images for given name on given step.

        Args:
            name (str): name of the variable (it will be converted to a valid
                tensorflow summary name).
            images (list): list of images to visualize
            step (int): non-negative integer used for visualization
        """
        if isinstance(images, six.string_types):
            raise TypeError('"images" should be a list of ndarrays, got {}'
                            .format(type(images)))

        self._check_step(step)
        tf_name = self._ensure_tf_name(name)

        summary = self._image_summary(tf_name, images, step=step)
        self._log_summary(tf_name, summary, images, step=step)

    def _image_summary(self, tf_name, images, step=None):
        """
        Log a list of images.

        References:
            https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py#L22

        Example:
            >>> tf_name = 'foo'
            >>> value = ([0, 1, 2, 3, 4, 5], [1, 20, 10, 22, 11])
            >>> self = Logger(None, is_dummy=True)
            >>> images = [np.random.rand(10, 10), np.random.rand(10, 10)]
            >>> summary = self._image_summary(tf_name, images, step=None)
            >>> assert len(summary.value) == 2
            >>> assert summary.value[0].image.width == 10
        """
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = summary_pb2.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1]
            )
            # Create a Summary value
            img_value = summary_pb2.Summary.Value(tag='{}/{}'.format(tf_name, i),
                                                  image=img_sum)
            img_summaries.append(img_value)
            summary = summary_pb2.Summary()
            summary.value.add(tag=tf_name, image=img_sum)

        summary = summary_pb2.Summary(value=img_summaries)
        return summary

    def _histogram_summary(self, tf_name, value, step=None):
        """
        Args:
            tf_name (str): name of tensorflow variable
            value (tuple or list): either a tuple of bin_edges and bincounts or
                a list of values to summarize in a histogram.

        References:
            https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py#L45

        Example:
            >>> tf_name = 'foo'
            >>> value = ([0, 1, 2, 3, 4, 5], [1, 20, 10, 22, 11])
            >>> self = Logger(None, is_dummy=True)
            >>> summary = self._histogram_summary(tf_name, value, step=None)
            >>> assert summary.value[0].histo.max == 5

        Example:
            >>> tf_name = 'foo'
            >>> value = [0.72,  0.18,  0.34,  0.66,  0.11,  0.70,  0.23]
            >>> self = Logger(None, is_dummy=True)
            >>> summary = self._histogram_summary(tf_name, value, step=None)
            >>> assert summary.value[0].histo.num == 7.0
        """
        if isinstance(value, tuple):
            bin_edges, bincounts = value
            assert len(bin_edges) == len(bincounts) + 1, (
                'must have one more edge than count')
            hist = summary_pb2.HistogramProto()
            hist.min = float(min(bin_edges))
            hist.max = float(max(bin_edges))
        else:
            values = np.array(value)

            bincounts, bin_edges = np.histogram(values)

            hist = summary_pb2.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values**2))

        # Add bin edges and counts
        for edge in bin_edges[1:]:
            hist.bucket_limit.append(edge)
        for v in bincounts:
            hist.bucket.append(v)

        summary = summary_pb2.Summary()
        summary.value.add(tag=tf_name, histo=hist)
        return summary

    def _scalar_summary(self, tf_name, value, step=None):
        summary = summary_pb2.Summary()
        summary.value.add(tag=tf_name, simple_value=value)
        return summary

    def _make_tf_name(self, name):
        tf_base_name = tf_name = make_valid_tf_name(name)
        i = 1
        while tf_name in self._tf_names:
            tf_name = '{}/{}'.format(tf_base_name, i)
            i += 1
        self._tf_names.add(tf_name)
        return tf_name

    def _log_summary(self, tf_name, summary, value, step=None):
        event = event_pb2.Event(wall_time=self._time(), summary=summary)
        if step is not None:
            event.step = int(step)
        if self.is_dummy:
            self.dummy_log[tf_name].append((step, value))
        else:
            self._write_event(event)

    def _write_event(self, event):
        data = event.SerializeToString()
        # See RecordWriter::WriteRecord from record_writer.cc
        w = self._writer.write
        header = struct.pack('Q', len(data))
        w(header)
        w(struct.pack('I', masked_crc32c(header)))
        w(data)
        w(struct.pack('I', masked_crc32c(data)))
        self._writer.flush()

    def _time(self):
        return self._dummy_time or time.time()

    def __del__(self):
        if self._writer is not None:
            self._writer.close()


def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)


def u32(x):
    return x & 0xffffffff


def make_valid_tf_name(name):
    if not _VALID_OP_NAME_START.match(name):
        # Must make it valid somehow, but don't want to remove stuff
        name = '.' + name
    return '_'.join(_VALID_OP_NAME_PART.findall(name))


_default_logger = None  # type: Logger


def configure(logdir, flush_secs=2):
    """ Configure logging: a file will be written to logdir, and flushed
    every flush_secs.
    """
    global _default_logger
    if _default_logger is not None:
        raise ValueError('default logger already configured')
    _default_logger = Logger(logdir, flush_secs=flush_secs)

def unconfigure():
    """ UnConfigure logging
    """
    global _default_logger
    _default_logger = None  # type: Logger

def _check_default_logger():
    if _default_logger is None:
        raise ValueError(
            'default logger is not configured. '
            'Call tensorboard_logger.configure(logdir), '
            'or use tensorboard_logger.Logger')


def log_value(name, value, step=None):
    _check_default_logger()
    _default_logger.log_value(name, value, step=step)


def log_histogram(name, value, step=None):
    _check_default_logger()
    _default_logger.log_histogram(name, value, step=step)


def log_images(name, images, step=None):
    _check_default_logger()
    _default_logger.log_images(name, images, step=step)

log_value.__doc__ = Logger.log_value.__doc__
