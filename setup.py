from setuptools import setup, find_packages

setup(
  name='PyToolBox',
  version='0.1.0',
  author='Zachary D Wallen',
  description="""A personal collection of convenience functions for styling data 
    displays and outputs, transforming and organizing data, performing statistical 
    analyses, and other commonly performed actions in data science projects.""",
  packages=find_packages(),
  install_requires=[
    'IPython',
    'numpy',
    'openpyxl',
    'pandas',
    'plotnine>=0.10.1',
    'scipy',
  ],
  python_requires='>=3.7',
)
