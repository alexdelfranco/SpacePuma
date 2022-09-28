from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="SpacePuma",
    version="0.14",
    author="Alex DelFranco",
    author_email="adelfranco24@amherst.edu",
    description=("Interact with your data."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexdelfranco/SpacePuma",
    packages=["spacepuma"],
    license="MIT",
    install_requires=[
        "numpy",
        "scipy",
        "jupyterlab",
        "matplotlib",
        "seaborn",
        "ipympl",
        ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
