# Project 1
## Setup

Make sure you have the following installed:

- miniconda
- $\LaTeX$
- Quarto
- (recommended) VSCode with extensions: _Quarto_, _Python_, _Python Environments_, and _Jupyter_

### Python Env Setup and Jupyter Kernel

Assuming `conda` is in your `PATH` and you trust me, run:

```shell
source setup.sh
```

Otherwise, please open the `setup.sh` file and go through the steps yourself. The comments prefixed with _[manual]_ show you the necessary commands.

## Running the Quarto Notebook in VSCode

### Running Individual Cells

When running individual cells, VSCode should ask you to "Select a kernel to run cells". Click on it and then select "Jupyter kernel...". Then select the kernel previously created "Python (p1)".

### Render the Whole Notebook

To run all code cells and render the whole notebook, select the preview icon at the top right corner or press `Cmd + Shift + K` (macOs), `Ctrl + Shift + K` (Linux/Windows). The correct jupyter kernel should automatically be selected because it is specified in the header of the Quarto file.