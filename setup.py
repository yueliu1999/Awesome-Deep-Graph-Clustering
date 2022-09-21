# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2022/9/19 20:52
import setuptools
from dgc.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep_graph_clustering",
    version="1.0.0",
    author="Yue Liu",
    author_email="yueliu1999@163.com",
    description="Awesome Deep Graph Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'munkres',
        'scikit_learn',
        'tqdm',
    ],
    py_requires=["dgc"],
    python_requires='>=3.6',
)
