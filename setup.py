from setuptools import setup

setup(
    name='SVSLoader',
    version='0.0.1',
    packages=['SVSLoader.Tests', 'SVSLoader.Config', 'SVSLoader.Utils', 'SVSLoader.Loaders', 'SVSLoader.Processing',
              'SVSLoader.InferenceUtils'],
    url='',
    license='',
    author='Benjamin Isaac Wilson',
    author_email='benjamintaya0111@gmail.com',
    description='SVS Whole Slide Image Loader and Processor - Various utilities to load large collections of SVS images'
                'and annotations files with an emphasis on automated file search and listing. Currently being developed'
                'for deep learning projects and experiments that require processing patch data.'
)
