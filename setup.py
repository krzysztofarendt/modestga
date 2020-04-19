from setuptools import setup

setup(name='modestga',
      version='0.2',
      description='Genetic Algorithm minimization',
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
          'numpy>=1.18.2',
          'matplotlib>=3.2.1',
          'scipy>=1.4.1'
      ],
      classifiers = [
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License'
      ]
      )
