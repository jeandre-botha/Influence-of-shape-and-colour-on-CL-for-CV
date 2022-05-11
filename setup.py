from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='Influence of shape and colour on curriculum learning for computer vision',
    version='1.0.0',
    description='Experimental software used to investigate the influence of shape and colour on curriculum learning for computer vision',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jeandre Botha',
    author_email='u17094446@tuks.co.za',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Research',
        'License :: MIT License :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='Curriculum Learning, Computer Vision, Convolutional Neural Network, Machine Learning',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8, <4",
    entry_points={
        "console_scripts": [
            "train=trainer:main",
            "test=tester:main",
        ],
    },
)
