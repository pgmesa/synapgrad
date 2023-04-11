import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt", "r", encoding='utf-8') as fh:
    req = fh.readlines()
    requirements = []
    for line in req:
        requirements.append(line.replace("\n", "")) # \ufeff

setuptools.setup(
    name='synapgrad',  
    version='0.3.0',
    author="Pablo García Mesa",
    author_email="pgmesa.sm@gmail.com",
    description="An autograd Tensor-based engine with a deep learning library built on top of it made from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pgmesa/synapgrad",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
 )