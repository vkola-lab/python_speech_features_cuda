"""
Created on Fri Jul 31 16:55:14 2020

@author: cxue2
"""

import setuptools

setuptools.setup(
    name='python_speech_features_cuda',
    version='0.0.12',
    author='Chonghua Xue',
    author_email='cxue2@bu.edu',
    url='https://github.com/vkola-lab/python_speech_features_cuda',
    description='Re-implementation of Python Speech Features Extraction on CUDA.',
    packages=setuptools.find_packages(),
    classifiers=[
        'Environment :: GPU :: NVIDIA CUDA',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ]
)