import os
import setuptools #type: ignore
from configparser import ConfigParser
from pkg_resources import parse_version #type: ignore

assert parse_version(setuptools.__version__) >= parse_version('36.2')


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
statuses = ['1 - Planning', '2 - Pre-Alpha', '3 - Alpha', '4 - Beta',
            '5 - Production/Stable', '6 - Mature', '7 - Inactive']

cfg_keys = 'description keywords author author_email lib_name license version url language'.split()
setup_cfg = {o: cfg[o] for o in cfg_keys}

with open(cfg.get('requirements','')) as f:
    requirements = f.readlines()

setuptools.setup(
    packages=setuptools.find_packages(),
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
    classifiers=['Development Status :: ' + statuses[int(cfg['status'])],
                 'Intended Audience :: ' + cfg['audience'].title(),
                 'License :: ' + cfg['license'],
                 'Natural Language :: ' + cfg['language'].title()] +
                ['Programming Language :: Python :: ' + o for o in
                 py_versions[py_versions.index(min_python):]],
    **setup_cfg)