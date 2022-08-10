from setuptools import find_packages, setup


setup(
    name = 'psg-simultscoring-models',
    version = '0.0.1dev0',
    author = 'Riku Huttunen',
    author_email = 'riku.huttunen@uef.fi',
    description = 'The models for simultaneous scoring of sleep stages and respiratory events with deep learning.',
    license = 'MIT',
    packages = find_packages(exclude='tests*'),
    install_requires = [
        'numpy',
        'tensorflow',
    ]
)
