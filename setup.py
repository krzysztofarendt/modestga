from setuptools import setup
import setuptools

setup(name='modestga',
      version='0.6',
      description='Genetic Algorithm minimization',
      url='https://github.com/krzysztofarendt/modestga',
      keywords='genetic algorithm optimization',
      author='Krzysztof Arendt',
      author_email='krzysztof.arendt@gmail.com',
      license='BSD',
      platforms=['Windows', 'Linux'],
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=[
          'numpy>=1.18.2',
          'matplotlib>=3.2.1',
          'scipy>=1.4.1',
          'cloudpickle'
      ],
      classifiers = [
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License'
      ]
  )
