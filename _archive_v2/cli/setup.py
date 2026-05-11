from setuptools import setup, find_packages

setup(
    name="company-brain-cli",
    version="1.0.0",
    description="CLI for Company Brain - structured knowledge layer for AI agents",
    packages=find_packages(),
    install_requires=[
        "click>=8.0",
        "requests>=2.28",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "brain=brain.main:cli",
        ],
    },
    python_requires=">=3.9",
)
