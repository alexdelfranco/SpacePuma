# AstroChem Tools

Tools developed for the Öberg Astrochemistry Group

The tools included are aimed at increasing the expediancy of data analysis and reduction within the fields of chemistry and astro-chemistry. Tutorial notebooks are included within the Tutorials directory. For detailed instructions on how to modify or add to the tools, check our documentation page [here]().

## Installation

The methods utilized in these tools require Python version 3.8.5 or above.

Upon cloning the repository, you will need to install the series of necessary packages using either the environment.yml file or the requirements.txt file. The recommended setup method is to use create a conda environment sourced from the environment.yml file. The environment can then be activated by following teh steps below. All code should then be executed within that activated environment. Alternatively, you can instruct pip to install all the necessary packages. If utilizing pip, it is recommended that the tools are run in a virtual python environment using pipenv. The environment should ensure that your depenency graph is compatible.

You can complete either installation method by running the following:

### To create a conda environment:

```bash
conda env create -f environment.yml
```

To activate the environment:

```bash
conda activate astrochem-tools
```

The environment should be activated in this manner each time the tools are to be used, and deactivated before other python code is run. To deactivate a conda environment, run either:

```bash
conda activate
```

or

```bash
conda activate base
```

Either of these deactivation methods are far superior to ```conda deactivate``` as per the conda user guide, if you run ```conda deactivate``` from your base environment, you may lose the ability to run conda at all.

### To install packages via pip with a virtual environment

```bash
# Install pipenv using pip
pip3 install pipenv
# Alternatively, if on MacOS, you can use HomeBrew
brew install pipenv
# From inside the repository, activate the environment shell
pipenv shell
```

### To install packages using pip with no virtual environment

This installation method is not recommended, as it could lead to version control issues.

```bash
pip install -r requirements.txt
```

Now, all the necessary packages should be installed!

## Tutorials

Once you have cloned the repository and setup all the necessary virtual environments, you can make use of some of the tutorials we have designed. To access the tutorials, enter the tutorials folder and launch the tutorial notebook:

```bash
cd Tutorials
jupyter notebook
```

Happy coding!
