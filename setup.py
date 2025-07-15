# coding=utf-8
from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nirapi",
    version="1.0.0",
    author="zata",
    description="A Near-Infrared Spectroscopy Analysis API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zata/nirapi",
    project_urls={
        "Bug Tracker": "https://github.com/zata/nirapi/issues",
        "Documentation": "https://github.com/zata/nirapi#readme",
        "Source Code": "https://github.com/zata/nirapi"
    },
    packages=find_packages(),
    package_data={
        'nirapi': ['*.pkl'],  # Include pickle files
    },
    include_package_data=True,
    python_requires=">=3.8",
    license="MIT",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "optuna",
        "pybaselines",
        "xgboost",
        "catboost", 
        "lightgbm",
        "obspy",
        "tqdm",
        "requests",
        "pymysql",
        "joblib",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="NIR spectroscopy analysis machine-learning",
)
