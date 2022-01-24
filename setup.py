from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="neuroxai",
    version="0.1",
    description="Interpretability Callbacks for Tensorflow 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ramy Zeineldin",
    author_email="ramy.zeineldin@student.kit.edu",
    url="https://github.com/razeineldin/neuroxai",
    license="MIT",
    install_requires=[],
    extras_require={
        "tests": [
            "black>=19.3b0",
            "pylint>=2.3.1",
            "pytest>=5.0.1",
            "pytest-timeout>=1.3.3",
            "pytest-mock>=1.10.4",
            "pytest-cov>=2.7.1",
            "tox>=3.13.2",
        ],
        "publish": ["bumpversion>=0.5.3", "twine>=1.13.0"],
        "docs": ["sphinx>=2.1.2", "sphinx-rtd-theme>=0.4.3"],
    },
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
