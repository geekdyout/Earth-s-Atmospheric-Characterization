from setuptools import setup, find_packages

setup(
    name="atmospheric-remote-sensing",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "xarray>=0.19.0",
        "netCDF4>=1.5.7",
        "h5py>=3.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "gdal>=3.3.0",
        "rasterio>=1.2.0",
        "dask>=2021.6.0",
        "distributed>=2021.6.0",
        "psutil>=5.8.0",
        "pyyaml>=5.4.0",
        "requests>=2.25.0",
        "boto3>=1.18.0",
        "earthdata>=0.1.0",
        "sentinelsat>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.2",
            "dvc>=2.6.0",
            "gitpython>=3.1.0",
            "tqdm>=4.62.0",
            "ipykernel>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
    author="Yadu Krishnan",
    author_email="yadutz3256@gmail.com",
    description="A framework for atmospheric remote sensing data analysis",
    keywords="remote sensing, atmospheric science, cloud detection, aerosol analysis",
    python_requires=">=3.8",
)