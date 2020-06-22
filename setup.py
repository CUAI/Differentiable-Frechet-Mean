import setuptools

setuptools.setup(
    name="frechetmean",
    version="0.0.1",
    author="Aaron Lou",
    author_email="al968@cornell.edu",
    description="Differentiable Frechet Mean",
    #url="https://github.com/CUVL/Differentiable-Frechet-Mean",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.5.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)