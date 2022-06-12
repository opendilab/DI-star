# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Module setuptools script."""

from setuptools import setup

description = """distar - StarCraft II Learning Environment

Part1 distar.ctools.pysc2:
ctools.pysc2 is DeepMind's Python component of the StarCraft II Learning Environment
(SC2LE). It exposes Blizzard Entertainment's StarCraft II Machine Learning API
as a Python RL Environment. This is a collaboration between DeepMind and
Blizzard to develop StarCraft II into a rich environment for RL research. distar.ctools.pysc2
provides an interface for RL agents to interact with StarCraft 2, getting
observations and sending actions.

We have published an accompanying blogpost and paper
https://deepmind.com/blog/deepmind-and-blizzard-open-starcraft-ii-ai-research-environment/
which outlines our motivation for using StarCraft II for DeepRL research, and
some initial research results using the environment.

Read the README at https://github.com/deepmind/ctools.pysc2 for more information.

Part2 TStarBot1:
Macro-action-based StarCraft-II learning environment.
"""

setup(
    name='distar',
    version='0.0.1',
    description='Starcraft II environment and library for training agents.',
    long_description=description,
    author='X-lab',
    license='Apache License, Version 2.0',
    keywords='StarCraft AI',
    packages=[
        'distar.pysc2',
        'distar.pysc2.agents',
        'distar.pysc2.bin',
        'distar.pysc2.env',
        'distar.pysc2.lib',
        'distar.pysc2.maps',
        'distar.pysc2.run_configs',
        'distar.pysc2.tests',
        'distar.actor',
        'distar',
        'distar.agent',
        'distar.bin',
        'distar.bin',
        'distar.envs',
        'distar.ctools',
        'distar.ctools.data',
        'distar.ctools.torch_utils',
        'distar.ctools.utils',
        'distar.ctools.worker'
    ],
    install_requires=[
        'absl-py>=0.1.0',
        'dill',
        'mock',
        'mpyq',
        'numpy>=1.10',
        'portpicker>=1.2.0',
        'protobuf<=3.20.1',
        'pygame',
        'requests',
        's2clientprotocol>=3.19.0.58400.0',
        'six',
        'sk-video',
        'websocket-client',
        'whichcraft',
        'gym',
        'joblib',
        'pyzmq',
        'sphinx',
        'sphinx_rtd_theme',
        'pyyaml',
        'easydict',
        'opencv-python',
        'tensorboardX',
        'tabulate',
        'matplotlib',
        'yapf==0.29.0',
        'flask',
        'lz4',
        'sc2reader',
        'pyarrow',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
