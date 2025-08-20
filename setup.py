from setuptools import setup, find_packages

setup(
    name="django-nl-filter",
    version="1.0",
    author="Arjun V S",
    author_email="arjunvs.vs@gmail.com",
    description="A natural language filter for Django ORM queries",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/arjun-vs/django-nl-filter",  # update if you have GitHub
    packages=find_packages(),
    include_package_data=True,   # picks files from MANIFEST.in
    install_requires=[
        "Django>=3.2",
    ],
    classifiers=[
        "Framework :: Django",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
