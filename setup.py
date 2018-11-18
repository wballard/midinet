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
          'keras==2.2.4',
          'tensorflow==1.12.0',
      ],
      scripts=[
          'bin/midinet',
      ],
      zip_safe=False)
