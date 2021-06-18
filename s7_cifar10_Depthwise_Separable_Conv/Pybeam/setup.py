import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beam-pkg",  # Replace with your own username
    version="0.0.1",
    author="Jenisha T, Vivek, Chetan",
    author_email="joyjeni@gmail.com, chethanv23@gmail.com, vivek.mail0205@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joyjeni/ComputerVision/tree/main/s7_cifar10_Depthwise_Separable_Conv/Pybeam",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
