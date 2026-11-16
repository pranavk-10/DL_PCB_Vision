# PCB-Vision-AI

This directory contains the Python environment and Git configuration for a PCB
vision project. No application, training pipeline, or dataset has been added yet.

## Windows PowerShell setup

Use **64-bit Python 3.11**. The detected interpreter is Python **3.11.9**.
The pinned dependencies in `requirements.txt` target this interpreter. Other
Python versions have not been tested; use Python 3.11 for this setup.

```powershell
cd C:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI
python --version
python -c "import sys; print(sys.executable); print('64-bit:', sys.maxsize > 2**32)"

# Preserve an existing environment. Create one only when its interpreter is absent.
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip --version
python -c "import sys; print(sys.executable); print(sys.prefix); print(sys.base_prefix)"
python scripts\verify_environment.py
```

After activation, `python` must resolve to `PCB-Vision-AI\.venv\Scripts\python.exe`
and pip must load from `PCB-Vision-AI\.venv\Lib\site-packages`. If activation did
not succeed, use the explicit environment interpreter for every command:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

Never use a global `pip install`, `--user`, or `--system-site-packages` for this
project. `pyvenv.cfg` must contain `include-system-site-packages = false`.
Activation applies to the current shell; activate again in each new terminal.
Exit the active environment with `deactivate`.

### PowerShell activation restrictions

If PowerShell reports that running scripts is disabled, allow locally generated
scripts for the current process only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
Get-Command python,pip | Select-Object Name,Source
python -m pip --version
```

If organization policy prevents that change, use the explicit `.venv` interpreter
commands above. No persistent machine execution-policy change is required.

### Dependencies and compatibility

| Distribution | Pinned version | Import |
| --- | --- | --- |
| TensorFlow | 2.20.0 | `tensorflow` |
| NumPy | 2.2.6 | `numpy` |
| OpenCV | 4.12.0.88 | `cv2` |
| Pandas | 2.3.3 | `pandas` |
| Matplotlib | 3.10.7 | `matplotlib` |
| Scikit-learn | 1.7.2 | `sklearn` |
| Streamlit | 1.50.0 | `streamlit` |
| Jupyter | 1.1.1 | `jupyter` |
| Notebook | 7.4.7 | `notebook` |
| IPython kernel | 6.30.1 | `ipykernel` |
| Pillow | 11.3.0 | `PIL` |

NumPy 2.2.6 is selected for the overlap between TensorFlow 2.20's NumPy
requirement and OpenCV 4.12's NumPy 2.x requirement (`>=2,<2.3` on Python 3.11).
Pip resolves transitive dependencies during installation. These pins are a
candidate set until installation, `pip check`, imports, and the interoperability
check all pass. Native Windows TensorFlow uses CPU execution; this setup does
not configure CUDA.

The verification script checks `.venv`, Python version and architecture,
interpreter and pip locations, exclusion of global and user packages, all imports
and direct dependency versions, and `pip check`. When all imports work, it also
runs a small TensorFlow/NumPy/OpenCV/Pandas/Scikit-learn/Matplotlib smoke test.
It exits with a nonzero status on failure. `pip check` alone cannot establish that
required packages are installed: it also passes in an otherwise empty environment.

### Jupyter and Streamlit

Launch Jupyter using this environment so notebooks use its installed kernel:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook
```

In an editor, select `.venv\Scripts\python.exe` as the notebook interpreter.
When a Streamlit application exists, run it using the same interpreter:

```powershell
# Replace path_to_app.py with the actual application file when it is added.
.\.venv\Scripts\python.exe -m streamlit run path_to_app.py
```

### Correcting setup failures

For an interrupted environment creation with missing pip or activation scripts,
repair the existing environment in place:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade --default-pip
python -m venv .venv --without-pip
```

The second command uses the base Python 3.11 interpreter to regenerate the venv
scripts while preserving the pip installation. Do not delete a working `.venv`.
If Python itself is missing, install 64-bit Python 3.11 and rerun
`python -m venv .venv` from this directory.


If `python -m pip --version` is local but `Get-Command pip` resolves outside
`.venv`, regenerate its launchers using Python's bundled pip wheel:

```powershell
$pcbPipWheel = .\.venv\Scripts\python.exe -c "import ensurepip,pathlib; print(next((pathlib.Path(ensurepip.__file__).parent / '_bundled').glob('pip-*.whl')))"
.\.venv\Scripts\python.exe -m pip install --no-index --force-reinstall $pcbPipWheel
.\.venv\Scripts\Activate.ps1
Get-Command python,pip | Select-Object Name,Source
```

If package downloads fail with `WinError 10013`, run these corrective commands
from a terminal with permitted access to `pypi.org` and `files.pythonhosted.org`:

```powershell
cd C:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

For an offline installation, on another **Windows x64 / Python 3.11** machine
with network access, copy `requirements.txt` and download wheels for the full
dependency closure and pip:

```powershell
python -m pip download --only-binary=:all: --dest C:\PCB-Wheels pip -r requirements.txt
```

Copy `C:\PCB-Wheels` to the target machine, then run inside this project:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-index --find-links C:\PCB-Wheels --upgrade pip
.\.venv\Scripts\python.exe -m pip install --no-index --find-links C:\PCB-Wheels -r requirements.txt
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

## Git configuration

`.gitignore` excludes virtual environments, secrets, caches, IDE and operating
system files, logs, temporary files, datasets, generated results, and common
trained-model binaries. It preserves README files within dataset and result
directories, including nested README files. Store datasets under `data/raw` or
`data/processed` and trained binaries under `models`. For TensorFlow SavedModel
exports, use a directory named `saved_model`. Binaries stored elsewhere require
an appropriate ignore rule before staging.

Commit `.gitignore`, `requirements.txt`, Python source, notebooks, README files,
documentation, and configuration that contains no secrets. Do not commit `.venv`,
datasets, trained model artifacts, generated figures/metrics, logs, or `.env`
files. No environment variables are currently needed, so no `.env.example` has
been created; the ignore rules allow one if the application later requires it.

Before initializing, check whether Git already manages this directory:

```powershell
git rev-parse --show-toplevel
```

Only when this reports that the directory is not a repository, initialize in the
project root. The requested commands are:

```powershell
git init
git status
git add .
git status
```

`git add .` stages files; it does not commit or push. It is provided for manual
use and was not run. Review the staged files before committing. For a directory
already inside a repository, skip `git init` and use the existing repository.
Do not reset or replace that repository or change its remotes.

Current repository discovery points to `C:/Users/hp/Desktop/Projects/DL`, the
existing parent repository. It is preserved; no nested repository is created.

Inspect ignore rules and trackable project files from this directory:

```powershell
git check-ignore -v .venv/Scripts/python.exe
git check-ignore -v --no-index --stdin
git ls-files --others --exclude-standard .
git status --short --untracked-files=all -- .
git ls-files -- .
```

For the `--stdin` command, enter candidate file paths one per line and end input
with Ctrl+Z, then Enter. Prefer `git add --dry-run .` to preview staging. Ignore
rules do not remove files that were already tracked; inspect `git ls-files`
before staging.

## Actual verification results

Measured on **2026-11-16**, using the project environment. **Environment setup
is incomplete: the virtual environment works, but project dependencies are not
installed and package compatibility has not been verified at runtime.**

| Check | Actual result |
| --- | --- |
| `.venv` exists | PASS; created with Python's built-in `venv` and repaired in place after an interrupted creation. |
| Python version and architecture | PASS; CPython 3.11.9, Windows x64. |
| PowerShell activation | PASS after setting `RemoteSigned` for the current process; initial activation was blocked by script policy. |
| `Get-Command python` | PASS; `PCB-Vision-AI\.venv\Scripts\python.exe`. |
| `Get-Command pip` | PASS after repairing the launcher; `PCB-Vision-AI\.venv\Scripts\pip.exe`. |
| Pip module/version | PASS; pip 24.0 from `.venv\Lib\site-packages\pip`. |
| Interpreter prefix | PASS; `sys.prefix` is this project's `.venv` and differs from `sys.base_prefix`. |
| Global/user package isolation | PASS; `include-system-site-packages = false` and `site.ENABLE_USER_SITE = False`. |
| Pip upgrade | Attempt exited 0 but left pip at bundled version 24.0; a newer version was not downloaded. |
| Required dependency installation | FAIL; pip exited 1 on `tensorflow==2.20.0`. |
| Required package imports | FAIL; all 11 imports produced `ModuleNotFoundError`; details below. |
| Pinned-version/runtime compatibility | NOT VERIFIED; dependencies are missing. |
| Scientific interoperability smoke test | SKIPPED because imports failed. |
| `pip check` | PASS, exit 0, `No broken requirements found.` This does not establish that required packages are installed. |
| Verifier result | FAIL; exit 1 with 11 failed import checks. |
| `.gitignore` exists | PASS. |
| Ignore/trackability rules | PASS; 70 candidate paths checked with `git check-ignore --no-index --stdin -z`, zero failures. |
| Actual `.venv` files ignored | PASS for `.venv/Scripts/python.exe` and `.venv/pyvenv.cfg`. |
| Dataset/model/generated-output paths ignored | PASS for representative paths, including nested data, model binaries, checkpoints, SavedModel assets, figures, and metrics. No datasets or trained models exist in this scaffold. |
| Source, notebooks, docs, config, README files trackable | PASS; representative source/notebook/docs/config paths and nested README paths are allowed; all nine actual project files are listed by Git as untracked and eligible for tracking. No application notebooks have been added. |
| Existing Git repository | PASS; parent repository at `C:/Users/hp/Desktop/Projects/DL` preserved. No nested `git init` needed. |
| `git status` / `git ls-files` | PASS; nine untracked project files, no tracked project files. |
| Staging preview (`git add --dry-run .`) | FAIL; parent `.git` is read-only in this execution environment; exact error below. |
| Commits, pushes, staging, remote changes | None performed. |

Actual import failures:

```text
FAIL import tensorflow: ModuleNotFoundError: No module named 'tensorflow'
FAIL import numpy: ModuleNotFoundError: No module named 'numpy'
FAIL import cv2: ModuleNotFoundError: No module named 'cv2'
FAIL import pandas: ModuleNotFoundError: No module named 'pandas'
FAIL import matplotlib: ModuleNotFoundError: No module named 'matplotlib'
FAIL import sklearn: ModuleNotFoundError: No module named 'sklearn'
FAIL import streamlit: ModuleNotFoundError: No module named 'streamlit'
FAIL import jupyter: ModuleNotFoundError: No module named 'jupyter'
FAIL import notebook: ModuleNotFoundError: No module named 'notebook'
FAIL import ipykernel: ModuleNotFoundError: No module named 'ipykernel'
FAIL import PIL: ModuleNotFoundError: No module named 'PIL'
```

The installation attempt used the activated project's interpreter:

```powershell
python -m pip install --upgrade pip --retries 0 --timeout 10 --disable-pip-version-check
python -m pip install -r requirements.txt --retries 0 --timeout 10 --disable-pip-version-check
```

Exact dependency-installation error:

```text
ERROR: Could not find a version that satisfies the requirement tensorflow==2.20.0 (from versions: none)
ERROR: No matching distribution found for tensorflow==2.20.0
Dependency installation exit code: 1
```

A separate request to `https://pypi.org/pypi/tensorflow/2.20.0/json` confirmed the
network restriction. The pip error therefore does not establish that this
TensorFlow release lacks a compatible wheel:

```text
URLError: <urlopen error [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions>
```

Use the network-enabled or offline installation commands above, then rerun the
verifier. Do not consider installation complete until all checks pass.

The interrupted creation initially reported `No module named pip` and was
missing `Activate.ps1`. Both were repaired without deleting `.venv`. The initial
activation error was:

```text
File C:\Users\hp\Desktop\Projects\DL\PCB-Vision-AI\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```

The process-scoped activation command above corrected this. The pip launcher
initially resolved globally; reinstalling Python's bundled pip wheel **inside
`.venv`** corrected it. No project packages were installed globally.

Exact Git staging-preview error:

```text
fatal: Unable to create 'C:/Users/hp/Desktop/Projects/DL/.git/index.lock': Permission denied
```

`git status`, `git ls-files`, and ignore validation succeeded despite this
restriction. To preview staging, run `git add --dry-run .` from this project in a
terminal with write access to the existing parent repository's `.git` directory.
The manual `git add .` command can then stage the reviewed files; no commit or
push is performed by this setup.
