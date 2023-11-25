from setuptools import setup, find_packages

setup(
    name='pyrl',
    version='0.0.1',
    author='Sean Fitzgerald',
    author_email='Fitzgerald.sean.t@gmail.com',
    description='A Python library for calculating rl training metrics',
    url='https://github.com/mrplants/pyrl',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
