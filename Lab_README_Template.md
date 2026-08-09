# COMP3742 Lab X Instructions - Oliver Wuttke (WUTT0019)
Ensure Python version 3.11 or later is installed on your system.

Change into the Source directory.
```bash
cd Source
```

Create and activate your virtual environment.
```bash
python -m venv .venv          # create
source .venv/bin/activate     # activate (Linux/macOS)
.venv\Scripts\activate        # activate (Windows)
```

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

To run any question file simply use:
```bash
python file.py
```

## Jupyter Setup
Ensure that previous steps are completed and the virtual environment is still active.

Install jupyter notebook or jupyter lab (choose one or the other):
```bash
pip install jupyterlab   # lab
pip install notebook     # notebook
```

Then simply run:
```bash
jupyter lab         # lab
jupyter notebook    # notebook
```