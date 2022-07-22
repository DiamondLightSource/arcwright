from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name="arcwright",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Tools for integrating data from the XPDF-ARC detector on I15-1",
    license="MIT",
    author="Dean Keeble",
    author_email="dean.keeble@diamond.ac.uk",
    url="https://github.com/DiamondLightSource/arcwright",
    packages=["arcwright"],
    entry_points={"console_scripts": ["arcwright=arcwright.cli:cli"]},
    install_requires=requirements,
    keywords="arcwright",
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
