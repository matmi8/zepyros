from setuptools import setup

setup(name='zepyros',
      version='0.1.2',
      description='ZErnike Polynomials analYsis of pROtein Shapes. A tool for image characterizarion.',
      url='',
      author='Mattia Miotto',
      author_email='miottomattia1@gmail.com',
      license='GPU',
      packages=['zepyros'],
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'scipy'
      ],
      zip_safe=False)
