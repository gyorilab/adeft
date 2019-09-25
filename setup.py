from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='adeft',
      version='0.4.0',
      description=('Acromine based Disambiguation of Entities From'
                   ' Text'),
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/indralab/adeft',
      download_url='https://github.com/indralab/adeft/archive/0.4.0.tar.gz',
      author='adeft developers, Harvard Medical School',
      author_email='albert_steppi@hms.harvard.edu',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      packages=find_packages(),
      install_requires=['nltk', 'scikit-learn>=0.20.0', 'wget',
                        'requests', 'flask'],
      extras_require={'test': ['nose', 'coverage']},
      keywords=['nlp', 'biology', 'disambiguation']
      )
