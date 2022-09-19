# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2022/9/19 20:52
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adgc",
    version="1.0",
    author="Yue Liu",
    author_email="yueliu1999@163.com",
    description="Awesome Deep Graph Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[''],
    py_requires=["./"],
    python_requires='>=3.6',
)
