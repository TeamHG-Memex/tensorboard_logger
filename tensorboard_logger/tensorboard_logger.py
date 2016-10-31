# -*- coding: utf-8 -*-
from collections import defaultdict
import re

import six
import tensorflow as tf


__all__ = ['Logger', 'configure', 'log_value']


_VALID_OP_NAME_START = re.compile('^[A-Za-z0-9.]')
_VALID_OP_NAME_PART = re.compile('[A-Za-z0-9_.\\-/]+')


class Logger(object):
    def __init__(self, logdir, flush_secs=2, is_dummy=False):
        self._session = tf.Session()
        if not is_dummy:
            self._writer = tf.train.SummaryWriter(logdir, flush_secs=flush_secs)
        self._loggers = {}
        self._tf_names = set()
        self.is_dummy = is_dummy
        if is_dummy:
            self.dummy_log = defaultdict(list)

    def log_value(self, name, value, step):
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

        if not isinstance(step, six.integer_types):
            raise TypeError('"step" should be an integer, got {}'
                            .format(type(step)))

        try:
            logger = self._loggers[name]
        except KeyError:
            tf_name = self._make_tf_name(name)
            logger = self._loggers[name] = self._make_logger(tf_name, value)
        logger(value, step)

    def _make_tf_name(self, name):
        tf_base_name = tf_name = make_valid_tf_name(name)
        i = 1
        while tf_name in self._tf_names:
            tf_name = '{}/{}'.format(tf_base_name, i)
            i += 1
        self._tf_names.add(tf_name)
        return tf_name

    def _make_logger(self, tf_name, value):
        dtype = tf.float32
        variable = tf.Variable(
            initial_value=value, dtype=dtype, trainable=False, name=tf_name)
        self._session.run(tf.initialize_variables([variable], tf_name))
        summary_op = tf.scalar_summary(tf_name, variable)
        new_value = tf.placeholder(dtype, shape=[])
        assign_op = tf.assign(variable, new_value)

        def logger(x, i):
            _, summary = self._session.run([assign_op, summary_op],
                                           {new_value: x})
            if self.is_dummy:
                self.dummy_log[tf_name].append((i, x))
            else:
                self._writer.add_summary(summary, i)

        return logger


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
