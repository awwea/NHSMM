import os
import setuptools
from configparser import ConfigParser


class CleanCommand(setuptools.Command):
    """Custom clean command to tidy up the project root."""
    user_options:list = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']
min_python = cfg['min_python']
py_versions = '3.5 3.6 3.7 3.8 3.9 3.10 3.11'.split()
statuses = ['1 - Planning', '2 - Pre-Alpha', '3 - Alpha', '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive']
cfg_keys = 'description keywords author author_email license version url'.split()
setup_cfg = {o: cfg[o] for o in cfg_keys}

with open('requirements.txt',) as f:
    requirements = f.readlines()

setuptools.setup(
    packages=setuptools.find_packages(exclude=('tests', 'docs')),
    test_suite='tests',
    install_requires=requirements,
    setup_requires=requirements,
    tests_require=requirements,
    python_requires='>=' + min_python,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    cmdclass={
        'clean': CleanCommand,
    },
    **setup_cfg)
