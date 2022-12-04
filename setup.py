"""
RAFT installer

To develop: python setup.py develop --user
"""
import setuptools


# Add other file types to package
DATA_FILES = [
    '*.json',
    '*.cl',
    '*.m',
    '*.c',
    '*.cpp',
    '*.h',
    '*.hpp',
    '*Makefile',
    'wscript',
    '*.jpg',
    '*.raw',
    '*.png',
]


PACKAGES = setuptools.find_packages()


SETUP = dict(
    name='raft',
    version='1.0',
    description='RAFT Optical Flow',
    packages=PACKAGES,
    package_data={pkg: DATA_FILES for pkg in PACKAGES},
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>2.6, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
)


if __name__ == '__main__':
    setuptools.setup(**SETUP)
