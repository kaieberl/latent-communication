from setuptools import setup, find_packages

setup(
    name='Latent Communication',
    version='0.1.0',
    url='https://github.com/kaieberl/latent-communication',
    author='',
    author_email='',
    description='Code base for Latent Communication case studies project',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchvision',
        'lightning',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'tensorboard',
        'pytorch-lightning',
        'seaborn',
    ],
)
