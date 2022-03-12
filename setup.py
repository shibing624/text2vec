# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('text2vec/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

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
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords='word embedding,text2vec,Chinese Text Similarity Calculation Tool,similarity,word2vec',
    install_requires=[
        "jieba>=0.39",
        "loguru",
        "transformers>=4.6.0",
        "datasets",
        "tqdm",
        "scikit-learn",
        "gensim>=4.0.0",
        "numpy",
        "pandas",
    ],
    packages=find_packages(exclude=['tests']),
    package_dir={'text2vec': 'text2vec'},
    package_data={'text2vec': ['*.*', 'data/*.txt']}
)
