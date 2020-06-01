from setuptools import setup, find_packages

setup(
  name = 'axial_attention',
  packages = find_packages(exclude=['examples']),
  version = '0.0.4',
  license='MIT',
  description = 'Axial Attention',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/axial-attention',
  keywords = ['attention', 'artificial intelligence'],
  install_requires=[
      'torch'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)