# -*- coding: utf-8 -*-
import re

import six
import tensorflow as tf


__all__ = ['Logger', 'configure', 'log_value']


class Logger(object):
    def __init__(self, logdir, flush_secs=2):
        self._session = tf.Session()
        self._writer = tf.train.SummaryWriter(logdir, flush_secs=flush_secs)
        self._loggers = {}

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

        # TODO - check that the name is unique
        name = '_'.join(re.findall('\w+', name))  # valid tf name
        try:
            logger = self._loggers[name]
        except KeyError:
            logger = self._loggers[name] = self._make_logger(name, value)
        logger(value, step)

    def _make_logger(self, name, value):
        dtype = tf.float32
        variable = tf.Variable(
            initial_value=value, dtype=dtype, trainable=False, name=name)
        self._session.run(tf.initialize_variables([variable], name))
        summary_op = tf.scalar_summary(name, variable)
        new_value = tf.placeholder(dtype, shape=[])
        assign_op = tf.assign(variable, new_value)

        def logger(x, i):
            _, summary = self._session.run([assign_op, summary_op],
                                           {new_value: x})
            self._writer.add_summary(summary, i)

        return logger


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
