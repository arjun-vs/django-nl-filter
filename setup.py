from setuptools import setup, find_packages

setup(
    name='django-nl-filter',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'ollama',
        'django',
    ],
    description='A dynamic, open-source package for converting natural language to advanced Django ORM queries via Ollama',
    author='Your Name',
    license='MIT',
)