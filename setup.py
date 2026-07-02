"""
Package configuration for the code/src library accompanying:

  N. Monyimba, "Quantitative Convergence Analysis of Stochastic Maximum
  Principles for Mean-Field Games with Common Noise" (MSc thesis, 2026),
  and the companion preprints in papers/.

Editable install:
    pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="applied-math-thesis",
    version="0.1.0",
    description=(
        "Numerical methods for common-noise mean-field games: "
        "Euler-Maruyama/Milstein SDE solvers and mean-field/particle "
        "variants with common-noise coupling."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Nisong Monyimba",
    author_email="nmonyimb@asu.edu",
    url="https://github.com/NisongMonyimba/AppliedMathThesis",
    license="MIT",
    package_dir={"": "code/src"},
    packages=find_packages(where="code/src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0",
        "scipy>=1.12",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "mypy>=2.0",
            "ruff>=0.6",
        ],
        "full": [
            "pandas>=2.2",
            "matplotlib>=3.8",
            "statsmodels>=0.14",
            "scikit-learn>=1.4",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
