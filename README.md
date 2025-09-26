# confocal-cell-seg

A tiny CLI to segment 3D confocal TIFF stacks, preview steps in Napari, and (optionally) export intermediate layers and centroids.

## Install

### With **uv** (recommended)

```bash
uv venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
uv pip install -e .
# (optional) dev extras
uv pip install -e ".[dev]"
```

### With **pip**

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e .
```

After install, the CLI `ccseg` is in your PATH.

## Run

### Batch over a folder

```bash
ccseg run --config config.yaml
# or
uv run ccseg run --config config.yaml
```

### Single file + visualization

```bash
ccseg single --config config.yaml --file "[PATH_TO_FILE]"
# or
uv run ccseg single --config config.yaml --file "[PATH_TO_FILE]"
```

### Convert Leica .lif → TIFF

```bash
ccseg convert --config config.yaml
# or
uv run ccseg convert --config config.yaml
```

**Outputs**

-   Centroids CSV → `data/out/centroids/<image>.csv`
-   Layer dumps (if enabled) → `data/out/<layer>/<image>.tif`
-   Napari opens if `visualization.enable: true`

## Configuration

The scripts are configured via a YAML file. See `config.yaml` for a full example.

## Notes

-   Expected TIFF layout: **C×Z×Y×X** or **Z×Y×X**. Others are skipped.
