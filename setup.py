from setuptools import setup
from setuptools import find_packages

setup(name='pytrade_env',
      version='0.1.0',
      description='backtest and livetrading for cryptocurrency',
      author='jjakimoto',
      author_email='f.j.akimoto@gmail.com',
      license='MIT',
      packages=find_packages(),
      py_modeuls=["pytrade_env", "test_pytrade_env"]
      )
