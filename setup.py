import os.path as osp
import pathlib
import pkg_resources
from setuptools import find_packages, setup


def fetch_requirements(name):
    if name not in ['runtime', 'tests', 'docs']:
        raise ValueError(f'Invalid name, should be one of [runtime, tests, docs], but got {name}')

    with pathlib.Path(f'requirements/{name}.txt').open() as requirements_txt:
        install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]
    try:
        _ = pkg_resources.get_distribution('mmcv-full')
        # mmcv-full is already installed,
        # so remove mmcv from the requirements list
        install_requires = [x for x in install_requires if 'mmcv' not in x]
    except pkg_resources.DistributionNotFound:
        pass

    return install_requires


def get_version():
    init_py_path = osp.join(osp.abspath(osp.dirname(__file__)), 'imba_explain', '__init__.py')
    init_py = open(init_py_path, 'r').readlines()
    version_line = [
        l.strip() for l in init_py  # noqa: E741
        if l.startswith('__version__')
    ][0]
    version = version_line.split('=')[-1].strip().strip("'\"")
    return version


packages = find_packages(exclude=['tests', 'tools'])

setup(
    name='imba-explain',
    url='',
    version=get_version(),
    author='',
    author_email='',
    license='MIT',
    description='TODO',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    install_requires=fetch_requirements('runtime'),
    packages=packages,
    extras_require={
        'tests': fetch_requirements('tests'),
        'docs': fetch_requirements('docs')
    },
    python_requires='>=3.7',
    keywords=['Deep Learning', 'Attribution', 'Explainable AI'],
)
