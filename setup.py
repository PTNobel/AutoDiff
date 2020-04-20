import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="auto_diff",
    version="0.3.1",
    author="Parth Nobel",
    author_email="parthnobel@berkeley.edu",
    description="An automatic differentiation library for Python+NumPy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PTNobel/autodiff",
    packages=setuptools.find_packages(),
    install_requires="numpy>=1.17",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
