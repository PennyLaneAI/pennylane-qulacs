# Copyright 2020 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
import re

from setuptools import setup

with open("./pennylane_qulacs/_version.py") as f:
    (version,) = re.findall('__version__ = "(.*)"', f.read())


requirements = [
    "pennylane>=0.42",
    "numpy",
    "scipy",
]

extra_requirements = {
    "cpu": ["qulacs>=0.1.10.1"],
    "gpu": ["qulacs-gpu>=0.1.10.1"],
}

info = {
    "name": "pennylane-qulacs",
    "version": version,
    "maintainer": "Xanadu Inc.",
    "maintainer_email": "software@xanadu.ai",
    "url": "http://xanadu.ai",
    "license": "Apache License 2.0",
    "packages": ["pennylane_qulacs"],
    "entry_points": {
        "pennylane.plugins": [
            "qulacs.simulator = pennylane_qulacs.qulacs_device:QulacsDevice"
        ]
    },
    "description": "PennyLane plugin for Qulacs.",
    "long_description": open("README.rst").read(),
    "long_description_content_type": "text/x-rst",
    "provides": ["pennylane_qulacs"],
    "install_requires": requirements,
    "extras_require": extra_requirements,
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    # Make sure to specify here the versions of Python supported
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **info)
