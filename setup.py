#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'six',
    'tensorflow',
]

test_requirements = [
    'pytest',
]

setup(
    name='tensorboard_logger',
    version='0.0.1',
    description='Log TensorBoard events without touching Tensorflow',
    long_description=readme + '\n\n' + history,
    author='Konstantin Lopuhin',
    author_email='kostia.lopuhin@gmail.com',
    url='https://github.com/TeamHG-Memex/tensorboard_logger',
    packages=[
        'tensorboard_logger',
    ],
    include_package_data=True,
    install_requires=requirements,
    license='MIT license',
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
