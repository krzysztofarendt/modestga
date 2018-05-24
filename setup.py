from setuptools import setup

setup(name='modestga',
      version='0.0.1',
      description='Genetic algorithm with a scipy-like interface',
      url='https://github.com/krzysztofarendt/modestga',
      keywords='genetic algorithm optimization',
      author='Krzysztof Arendt',
      author_email='krzysztof.arendt@gmail.com',
      license='BSD',
      platforms=['Windows', 'Linux'],
      packages=[
          'modestga'
      ],
      include_package_data=True,
      install_requires=[
          'numpy'
      ],
      classifiers = [
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License'
      ]
      )
