# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import re
import socket
import struct
import time

import six

from .tf_protobuf import summary_pb2, event_pb2
from .crc32c import crc32c


__all__ = ['Logger', 'configure', 'log_value']


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

    def log_value(self, name, value, step=None):
        """ Log new value for given name on given step.
        value should be a real number (it will be converted to float),
        and name should be a string (it will be converted to a valid
        tensorflow summary name). Step should be an non-negative integer,
        and is used for visualization: you can log several different
        variables on one step, but should not log different values
        of the same variable on the same step (this is not checked).
        """
        if not isinstance(name, six.string_types):
            raise TypeError('"name" should be a string, got {}'
                            .format(type(name)))

        if isinstance(value, six.string_types):
            raise TypeError('"value" should be a number, got {}'
                            .format(type(value)))
        value = float(value)

        if step is not None and not isinstance(step, six.integer_types):
            raise TypeError('"step" should be an integer, got {}'
                            .format(type(step)))

        try:
            tf_name = self._name_to_tf_name[name]
        except KeyError:
            tf_name = self._make_tf_name(name)
            self._name_to_tf_name[name] = tf_name

        self._log_value(tf_name, value, step)

    def _make_tf_name(self, name):
        tf_base_name = tf_name = make_valid_tf_name(name)
        i = 1
        while tf_name in self._tf_names:
            tf_name = '{}/{}'.format(tf_base_name, i)
            i += 1
        self._tf_names.add(tf_name)
        return tf_name

    def _log_value(self, tf_name, value, step=None):
        summary = summary_pb2.Summary()
        summary.value.add(tag=tf_name, simple_value=value)
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


def log_value(name, value, step=None):
    if _default_logger is None:
        raise ValueError(
            'default logger is not configured. '
            'Call tensorboard_logger.configure(logdir), '
            'or use tensorboard_logger.Logger')
    _default_logger.log_value(name, value, step=step)

log_value.__doc__ = Logger.log_value.__doc__
