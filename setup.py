from setuptools import setup

setup(name='midinet',
      version='0.1',
      description='Generate AI songs',
      url='http://github.com/wballard/midinet',
      author='Will Ballard',
      author_email='wballard@mailframe.net',
      license='MIT',
      packages=['midinet'],
      install_requires=[
          'tqdm',
          'mido',
          'docopt',
      ],
      scripts=[
          'bin/midinet',
      ],
      zip_safe=False)