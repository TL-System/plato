# Installation

## Setting up your Python environment

It is recommended that [Miniforge](https://github.com/conda-forge/miniforge) is used to manage Python packages. Before using *Plato*, first install Miniforge, update your `conda` environment, and then create a new `conda` environment with Python 3.9 using the command:

```shell
conda update conda -y
conda create -n plato -y python=3.9
conda activate plato
```

where `plato` is the preferred name of your new environment.

The next step is to install the required Python packages. PyTorch should be installed following the advice of its [getting started website](https://pytorch.org/get-started/locally/). The typical command in Linux with CUDA GPU support, for example, would be:

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117
```

The CUDA version, used in the command above, can be obtained on Ubuntu Linux systems by using the command:

```shell
nvidia-smi
```

Although PyTorch 2.0 will mostly likely work with Plato, it has not been as thoroughly tested as PyTorch 1.13.1 yet.

In macOS (without GPU support), the recommended command would be:

```shell
pip install torch==1.13.1 torchvision==0.14.1
```

## Installing Plato as a pip package

To use *Plato* as a Python framework, you only need to install it as a pip package:

```shell
pip install plato-learn
```

After *Plato* is installed, you can try to run any of the examples in `examples/`.

## Installing Plato for development

If you wish to modify the source code in *Plato* (rather than just using it as a framework), first clone this repository to a desired directory.

We will need to install several packages using `pip` as well:

```shell
pip install -r requirements.txt --upgrade
```

We will need to install both [PyLint](https://en.wikipedia.org/wiki/Pylint) and [Black](https://github.com/psf/black) (the official Python formatter in Plato):

```shell
pip install black pylint
```

Finally, we will install the current GitHub version of *Plato* as a local pip package:

```shell
pip install .
```

````{tip}

After the initial installation of the required Python packages, use the following command to upgrade all the installed packages at any time:

```shell
python upgrade_packages.py
```

If you are using a M1 or M2 Mac computer, a handy way to install [Miniforge](https://github.com/conda-forge/miniforge) is to do it using the command:

```shell
brew install miniforge
```

On M1 or M2 Mac computers, before installing the required packages in the conda environment, you may need to install the [Rust compiler](https://www.rust-lang.org/tools/install) first in order to install the `tokenizers` package:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

or simply

```shell
brew install rust
```

If you use Visual Studio Code, it is recommended to use `black` to reformat the code every time it is saved by adding the following settings to .`.vscode/settings.json`:

```
"python.formatting.provider": "black", 
"editor.formatOnSave": true
```

In general, the following is the recommended starting point for `.vscode/settings.json`:

```
{
	"editor.formatOnSave": true,
	"workbench.editor.enablePreview": false
}
```

When working in Visual Studio Code as your development environment, two of our colour theme favourites are called `Bluloco` (both of its light and dark variants) and `City Lights` (dark). They are both excellent and very thoughtfully designed.

The `Black Formatter`, `PyLint`, and `Python` extensions are required to be installed in Visual Studio Code.

````
