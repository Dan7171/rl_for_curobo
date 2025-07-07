"""
Setup script for RL CuRobo project.
"""

from setuptools import setup, find_packages

setup(
    name="rl_for_curobo",
    version="0.1.0",
    description="RL Module for CuRobo",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        # "torch",
        # "numpy",
        # Add other dependencies as needed
    ],
    python_requires=">=3.8",
)