# Contributing

Arcwright uses `hatch` for project management. Use these instructions to set up a development environment.

## Setting up a development conda environment
```
mkdir arcwright-dev
cd arcwright-dev
git clone git@github.com:DiamondLightSource/arcwright.git
cd arcwright
conda env create -n arc_dev_env --file CONTRIBUTING.yml
conda activate arc_dev_env
pre-commit install
```

## Using the environment
To enter a environment with arcwright installed in development mode (i.e. `pip install -e`)
```
conda activate arc_dev_env
cd arcwright-dev/arcwright
hatch shell
```
To exit the environment:
```
exit
```

## Versioning
arcwright uses git tags to track the version. Having made some commits, add a tag with the new version number i.e. `git tag v1.2.3`. Then push the tags `git push --tags` before running `hatch build` and `hatch publish`.

## Other supported features
| Command | What it does |
| --- | --- |
| `hatch run testing:cov` | Run the tests |
| `hatch run docs:serve` | Compile the documentation and serve locally for inspection |
| `hatch run docs:deploy` | Deploy the docs to github pages |
| `hatch build` | Perform the main build action |
| `hatch publish` | Publish artefacts to pypi |
