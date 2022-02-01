"""Module setup."""

import runpy
from setuptools import setup, find_packages

package_name = "segway-gbr"

version_meta = runpy.run_path("./segway_gbr/version.py")
version = version_meta["__version__"]

with open("README.md", "r") as fh:
        long_description = fh.read()

def parse_requirements(filename):
        """Load requirements from a pip requirements file."""
        lineiter = (line.strip() for line in open(filename))
        return [line for line in lineiter if line and not line.startswith("#")]

def main():
    setup(
            version=version,
            name=package_name,
            author="Mariam Arab",
            packages=find_packages(),
            install_requires=parse_requirements("requirements.txt"),
            python_requires=">=3.6.3",
            long_description=long_description,
            long_description_content_type="text/markdown",
            entry_points={'console_scripts': ['segway-gbr=segway_gbr.run:main']}
    )

if __name__ == "__main__":
    main()