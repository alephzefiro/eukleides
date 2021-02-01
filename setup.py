""" It's the setup. """
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='eukleides',
    version='0.0.1',
    description='Euclidean geometry related objects and algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alessandro Gentile',
    author_email='alemaudit@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='geometry',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6, <4',
    install_requires=['numpy>=1.0'],
    extras_require={
        'dev': [
            'pylint>=2.0.0',
            'black>=19.10b0',
            'mypy>=0.782',
            'pytest>=6.2.1',
        ],
    },
)
