import os
import sys
import platform
import efficientunet
from setuptools import setup

# "setup.py publish" shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist')
    os.system('twine upload dist/*')
    if platform.system() == 'Windows':
        os.system('powershell rm –path dist, efficientunet.egg-info –recurse –force')
    else:
        os.system('rm -rf dist efficientunet.egg-info')
    sys.exit()

setup(
    name='efficientunet',
    version=efficientunet.__version__,
    description="Keras Implementation of Unet with EfficientNet as encoder.",
    keywords='unet-keras',
    url='https://github.com/zhoudaxia233/efficientunet',
    license='MIT',
    packages=['efficientunet'],
    include_package_data=True,
    zip_safe=False,
    install_requires=['Keras>=2.2.4'],
    python_requires='>=3.6'
)
