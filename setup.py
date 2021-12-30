# -*- coding: utf-8 -*-
from __future__ import print_function

import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('text2vec/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    reqs = f.read()

setup(
    name='text2vec',
    version=__version__,
    description='Text to vector Tool, encode text',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/text2vec',
    license="Apache License 2.0",
    zip_safe=False,
    python_requires='>=3.5',
    classifiers=[
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='word embedding,text2vec,Chinese Text Similarity Calculation Tool,similarity,word2vec',
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(exclude=['tests']),
    package_dir={'text2vec': 'text2vec'},
    package_data={'text2vec': ['*.*', '../LICENSE', '../README.*', '../*.txt', 'data/*', 'utils/*']}
)
