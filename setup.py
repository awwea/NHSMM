from setuptools import setup, find_packages #type: ignore

with open('README.md', encoding="utf-8") as handle:
    LONG_DESCRIPTION = handle.read()

setup(
    name='chadhmm',
    packages=find_packages(),
    version='0.3.3',
    description='Package for Hidden (Semi) Markov Models',
    author='GarroshIcecream',
    author_email='ad.pesek13@gmail.com',
    url='https://github.com/GarroshIcecream/ChadHMM',
    download_url = 'https://github.com/GarroshIcecream/ChadHMM/archive/refs/tags/v0.3.tar.gz',
    license='MIT',
    keywords=['Hidden Markov Models','Hidden Semi-Markov Models','hsmm','hmm','Gaussian Mixture Models', 'gmm'],
    install_requires=['torch','matplotlib','scikit-learn','numpy','prettytable'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
    ]
)