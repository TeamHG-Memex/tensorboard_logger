# -*- coding: utf-8 -*-
from collections import defaultdict
import os
import re
import struct
import time
import zlib

import six
from .tf_protobuf import summary_pb2, event_pb2


__all__ = ['Logger', 'configure', 'log_value']


_VALID_OP_NAME_START = re.compile('^[A-Za-z0-9.]')
_VALID_OP_NAME_PART = re.compile('[A-Za-z0-9_.\\-/]+')


class Logger(object):
    def __init__(self, logdir, flush_secs=2, is_dummy=False):
        self._name_to_tf_name = {}
        self._tf_names = set()
        self.is_dummy = is_dummy
        self.logdir = logdir
        self.flush_secs = flush_secs  # TODO
        self._writer = None
        if is_dummy:
            self.dummy_log = defaultdict(list)
        else:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            filename = os.path.join(
                self.logdir,
                'events.out.tfevents.{}.ws'.format(int(time.time())))
            self._writer = open(filename, 'wb')
            self._write_event(event_pb2.Event(
                wall_time=_time, step=0, file_version='brain.Event:2'))

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
        event = event_pb2.Event(wall_time=_time, summary=summary)
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
        w(struct.pack('I', masked_crc32(header)))
        w(data)
        w(struct.pack('I', masked_crc32(data)))
        self._writer.flush()  # FIXME

    def __del__(self):
        if self._writer is not None:
            self._writer.close()


_time = 12.25


def masked_crc32(data):
    x = u32(zlib.crc32(data))
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


def log_value(name, value, step):
    if _default_logger is None:
        raise ValueError(
            'default logger is not configured. '
            'Call tensorboard_logger.configure(logdir), '
            'or use tensorboard_logger.Logger')
    _default_logger.log_value(name, value, step)

log_value.__doc__ = Logger.log_value.__doc__
