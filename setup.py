# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PetriDishNas",
    version="0.2",
    author="Shital Shah, Debadeepta Dey,",
    author_email="shitals@microsoft.com, dedey@microsoft.com",
    description="Implementation of Efficient Forward Architecture Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/petridishnn",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=[
        'warmup-scheduler @ git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git@v0.2',
        'pystopwatch2 @ git+https://github.com/ildoonet/pystopwatch2.git',
        'hyperopt', #  @ git+https://github.com/hyperopt/hyperopt.git
        'pretrainedmodels', 'tqdm', 'tensorboardx', 'sklearn', 'ray', 'matplotlib', 'psutil',
        'requests', 'tensorwatch', 'gorilla', 'pyyaml'
    ]
)

