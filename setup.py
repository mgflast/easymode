from setuptools import setup, find_packages
import shutil
# how to release:
# UPDATE VERSION IN 3 PLACES: Ais/core/config.py, setup.py, docs/conf.py

# push to pypi:
# python setup.py sdist
# twine upload dist/*

VERSION = '0.0.1'

setup(
    name='easymode',
    version=VERSION,
    packages=find_packages(),
    entry_points={'console_scripts': ['easymode=easymode.main:main']},
    url='',
    license='GPL v3',
    author='mgflast',
    author_email='mgflast@gmail.com',
    description='Easymode - a collection of pretrained general networks for segmenting common eukaryotic features in cryoET',
    package_data={'': ['*.png', '*.glsl', '*.pdf', '*.txt']},
    include_package_data=False,
    install_requires=[
        "tensorflow>=2.8.0,<3.0.0",
        "mrcfile",
        "numpy",
        "scipy",
        "huggingface_hub",
        "requests",
        "tifffile",
        "psutil",
        "starfile"
        #"Ais-cryoET"
    ]
)

