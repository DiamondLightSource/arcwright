# Welcome to arcwright

The tool for extracting diffraction patterns from 2D detectors which rotate about a sample.
Heavily based on [pyFAI](https://pyfai.readthedocs.io/en/master/).

## Commands

* `arcwright myfile.nxs` - Integrate the data in myfile
or use within your python application
```
from arcwright import arcwright
arcfai = arcwright.ArcFAI()
```
## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
