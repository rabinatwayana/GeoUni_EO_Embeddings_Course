# Setting up your machine for the hands-on exercises

Everything runs on a normal laptop (CPU only, 8 GB RAM is enough) with Python 3.11.
Tested on macOS (Apple Silicon) and Linux. Windows works via WSL2.

## 1. Install Python 3.11 and `uv`

`uv` is a fast Python package manager. Install it once:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Create the course environment

From the folder where you unpacked the course materials:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

This takes 2-5 minutes and downloads about 1.5 GB (PyTorch is the big one).

### Alternative: conda

```bash
conda create -n eo-embeddings python=3.11
conda activate eo-embeddings
pip install -r requirements.txt
```

## 3. Register the kernel and start JupyterLab

```bash
python -m ipykernel install --user --name eo-embeddings --display-name "EO embeddings"
jupyter lab notebooks/
```

In JupyterLab pick the **EO embeddings** kernel (top right of the notebook) if it is not already selected.

Open `01_first_embeddings.ipynb` and run the first cell. If it prints the library
versions without errors, you are ready.

## 4. Data and network

- The notebooks read embeddings directly from public cloud buckets and Hugging Face.
  No account or API key is needed for any exercise.
- Typical downloads per notebook: 50-200 MB.
- Notebook 6 (text search over Major TOM) runs entirely from files shipped with the course.
  Two optional cells download more: the SigLIP model (3.5 GB) to type your own text queries,
  and two 1 GB Major TOM files to cover northern Greece.
- Notebook 7 (Clay) downloads the 5.2 GB model checkpoint once and needs about 8 GB of free
  RAM while loading it (16 GB machines are comfortable, 8 GB machines should close other apps).
- Downloads are cached under `notebooks/data/cache/` and `~/.cache/`.

## 5. Common problems

| Symptom | Fix |
|---|---|
| `rasterio` fails to read `https://storage.googleapis.com/...` | Corporate proxy or firewall. Try from another network, or download the tile with your browser and point the notebook at the local file (each notebook shows how). |
| Very slow AlphaEarth reads (minutes) | Make sure the notebook's first cell ran: it sets `GDAL_NUM_THREADS=ALL_CPUS`, which cuts read time roughly tenfold. |
| `MemoryError` in notebook 4 or 5 | Reduce the area of interest, or increase `SUBSAMPLE` in the notebook. |
| `geotessera` reports no tiles | TESSERA coverage outside Europe/US is 2024 only. Check https://github.com/ucam-eo/tessera-coverage-map |
| Notebook 7 seems stuck while loading Clay (no CPU activity for minutes) | The 5 GB checkpoint load needs ~8 GB of free RAM; with less, the machine swaps and the kernel can stall. Close other notebooks and applications, restart the kernel, run again. |
| Notebook 5 takes long at 'nine years at 20 m' | Expected: 5-8 minutes of network reads. Start the cell and keep reading. |
