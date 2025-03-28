from setuptools import setup, find_packages

setup(
    name="vcl",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "jax>=0.4.13",
        "jaxlib>=0.4.13",
        "flax>=0.7.2",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    author="Jannis Buescher",
    author_email="definitelyjannisbuescher@gmail.com",
    description="A reimplementation of Variational Continual Learning (VCL)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/VCL_reimplementation",
) 