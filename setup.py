from setuptools import setup, find_packages
import retvid

setup(name='retvid',
      version=retvid.__version__,
      description='Demonstration of retinal transfer',
      author='George Sivulka',
      author_email='gsivulka@stanford.edu',
      url='https://github.com/gsivulka/retvid.git',
      install_requires=[i.strip() for i in open("requirements.txt").readlines()],
      long_description='''
          
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      packages=find_packages(),
)
