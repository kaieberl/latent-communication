from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [ln.split("#")[0].rstrip() for ln in f.readlines()]

setup(
    name='Latent Communication',
    version='0.1.0',
    url='https://github.com/kaieberl/latent-communication',
    author='',
    author_email='',
    description='Code base for Latent Communication case studies project',
    packages=find_packages(),
    install_requires=requirements
)
