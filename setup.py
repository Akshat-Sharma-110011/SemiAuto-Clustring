from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semiauto-clustering",
    version="1.0.3",
    author="Akshat-Sharma-110011",
    author_email="akshatsharma.business.1310@gmail.com",
    description="Automated clustering pipeline for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Akshat-Sharma-110011/SemiAuto-Clustring",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0,<1.26.0',
        'pandas',
        'scikit-learn',
        'scipy',
        'PyYAML',
        'cloudpickle',
        'fastapi',
        'uvicorn',
        'matplotlib',
        'seaborn',
        'optuna',
        'hdbscan'
    ],
    entry_points={
        'console_scripts': [
            'semiauto-clustering = semiauto_clustering.app:main',
        ],
    },
    include_package_data=True,
)