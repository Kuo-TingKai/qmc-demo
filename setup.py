from setuptools import setup, find_packages

setup(
    name="quantum-monte-carlo",
    version="0.1.0",
    description="Quantum Monte Carlo simulation with latest techniques",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.62.0",
    ],
    python_requires=">=3.8",
)

