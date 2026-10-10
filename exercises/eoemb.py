"""Small helper library for the *Earth Observation Embeddings* course notebooks.

Everything here is plain NumPy / rasterio / DuckDB. Read it: it is short on
purpose, and every hands-on explains the parts it uses.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform as warp_transform, transform_bounds
from rasterio.windows import Window

log = logging.getLogger("eoemb")

DATA_DIR = Path(__file__).resolve().parent / "data"
AEF_BUCKET_HTTPS = "https://storage.googleapis.com/alphaearth_foundations"
AEF_INDEX_URL = f"{AEF_BUCKET_HTTPS}/satellite_embedding/v1/annual/aef_index.parquet"
AEF_NODATA = -128

# GDAL settings that make windowed reads of remote COGs fast (measured: 90 s -> 8 s).
GDAL_REMOTE_OPTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_NUM_THREADS": "ALL_CPUS",
    "CPL_VSIL_CURL_CHUNK_SIZE": str(10 * 1024 * 1024),
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
}

BBox = tuple[float, float, float, float]  # (west, south, east, north) in EPSG:4326


def _check_bbox(bbox: BBox) -> BBox:
    w, s, e, n = map(float, bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"bbox must be (west, south, east, north) in degrees, got {bbox}")
    return w, s, e, n


# --------------------------------------------------------------------------- AlphaEarth
def load_aef_index(path: str | Path | None = None, bbox: BBox | None = None):
    """Load the official AlphaEarth tile index as a pandas DataFrame.

    `path` defaults to the Greece subset shipped with the course. Pass
    `path=eoemb.AEF_INDEX_URL` to query the full 300k-row global index straight
    from the bucket (DuckDB reads only the row groups it needs).
    """
    import duckdb

    src = str(path or DATA_DIR / "aef_index_greece.parquet")
    con = duckdb.connect()
    if src.startswith("http"):
        con.execute("INSTALL httpfs; LOAD httpfs;")
    where = ""
    if bbox is not None:
        w, s, e, n = _check_bbox(bbox)
        where = f"WHERE wgs84_east >= {w} AND wgs84_west <= {e} AND wgs84_north >= {s} AND wgs84_south <= {n}"
    cols = "path, year, utm_zone, crs, utm_west, utm_south, utm_east, utm_north, wgs84_west, wgs84_south, wgs84_east, wgs84_north"
    return con.execute(f"SELECT {cols} FROM read_parquet('{src}') {where} ORDER BY year, path").df()


def aef_tiles_for(index, bbox: BBox, year: int):
    """Rows of the index whose footprint intersects `bbox` for `year`, restricted to
    the UTM zone that covers the bbox centre (tiles overlap across zone borders)."""
    w, s, e, n = _check_bbox(bbox)
    sel = index[(index.year == year) & (index.wgs84_east >= w) & (index.wgs84_west <= e)
                & (index.wgs84_north >= s) & (index.wgs84_south <= n)]
    if sel.empty:
        raise LookupError(f"No AlphaEarth tile covers {bbox} in {year}. Is the index subset large enough?")
    zone_num = int((((w + e) / 2) + 180) // 6) + 1
    hemi = "N" if (s + n) / 2 >= 0 else "S"
    zone = f"{zone_num}{hemi}"
    in_zone = sel[sel.utm_zone == zone]
    return in_zone if not in_zone.empty else sel[sel.utm_zone == sel.utm_zone.iloc[0]]


def _read_window_north_up(ds, bounds_crs: tuple[float, float, float, float], factor: int = 1):
    """Read the window covering `bounds_crs` (in ds.crs), optionally decimated by an
    integer `factor` (served from the COG overviews, so 8 means 80 m pixels and ~1/64 of
    the bytes). Returns (array, transform) oriented north-up, even when the file stores
    rows south-to-north (AlphaEarth does)."""
    x0, y0, x1, y1 = bounds_crs
    r0, c0 = ds.index(x0, y0)
    r1, c1 = ds.index(x1, y1)
    rmin, rmax = sorted((r0, r1))
    cmin, cmax = sorted((c0, c1))
    rmin, cmin = max(rmin, 0), max(cmin, 0)
    rmax, cmax = min(rmax, ds.height), min(cmax, ds.width)
    if factor > 1:  # snap to the decimated grid so neighbouring tiles line up
        rmin, cmin = rmin - rmin % factor, cmin - cmin % factor
        rmax, cmax = rmax - rmax % factor, cmax - cmax % factor
    if rmax <= rmin or cmax <= cmin:
        return None, None
    win = Window(cmin, rmin, cmax - cmin, rmax - rmin)
    out_shape = (ds.count, (rmax - rmin) // factor, (cmax - cmin) // factor)
    arr = ds.read(window=win, out_shape=out_shape)
    tr = ds.window_transform(win)
    tr = Affine(tr.a * factor, tr.b, tr.c, tr.d, tr.e * factor, tr.f)
    if tr.e > 0:  # rows stored south -> north: flip to the usual north-up convention
        arr = arr[:, ::-1, :]
        tr = Affine(tr.a, tr.b, tr.c, tr.d, -tr.e, tr.f + tr.e * arr.shape[1])
    return arr, tr


def aef_read_bbox(bbox: BBox, year: int, index=None, factor: int = 1) -> tuple[np.ndarray, Affine, CRS]:
    """Read AlphaEarth annual embeddings for `bbox` straight from the public bucket.

    Returns (int8 array of shape (64, H, W), affine transform, CRS). Values are the
    raw quantized integers; -128 is nodata. Call `dequantize` to get floats.
    Tiles in the same UTM zone are mosaicked; the bbox must not span two zones.
    `factor` > 1 reads the COG overviews instead of full resolution (8 -> 80 m pixels):
    each overview pixel is the renormalised mean embedding of the pixels under it.
    """
    if factor < 1 or factor & (factor - 1):
        raise ValueError("factor must be a power of two (1, 2, 4, 8, ...)")
    w, s, e, n = _check_bbox(bbox)
    if index is None:
        index = load_aef_index()
    tiles = aef_tiles_for(index, bbox, year)
    crs = CRS.from_string(tiles.crs.iloc[0])
    bounds_utm = transform_bounds("EPSG:4326", crs, w, s, e, n)
    pieces = []
    with rasterio.Env(**GDAL_REMOTE_OPTS):
        for path in tiles.path:
            url = path.replace("gs://", "https://storage.googleapis.com/")
            with rasterio.open(url) as ds:
                arr, tr = _read_window_north_up(ds, bounds_utm, factor)
                if arr is not None:
                    pieces.append((arr, tr))
                    log.info("read %s -> %s", url.rsplit("/", 1)[-1], arr.shape)
    if not pieces:
        raise LookupError(f"AlphaEarth tiles for {year} do not overlap {bbox}")
    if len(pieces) == 1:
        return pieces[0][0], pieces[0][1], crs
    # Mosaic: all pieces share the 10 m grid of one UTM zone, so paste by pixel offset.
    res = pieces[0][1].a
    left = min(t.c for _, t in pieces)
    top = max(t.f for _, t in pieces)
    right = max(t.c + a.shape[2] * res for a, t in pieces)
    bottom = min(t.f - a.shape[1] * res for a, t in pieces)
    H, W = int(round((top - bottom) / res)), int(round((right - left) / res))
    canvas = np.full((pieces[0][0].shape[0], H, W), AEF_NODATA, dtype=np.int8)
    for a, t in pieces:
        r, c = int(round((top - t.f) / res)), int(round((t.c - left) / res))
        canvas[:, r:r + a.shape[1], c:c + a.shape[2]] = a
    return canvas, Affine(res, 0, left, 0, -res, top), crs


def dequantize(arr: np.ndarray) -> np.ndarray:
    """AlphaEarth int8 -> float32 in [-1, 1]: sign(v) * (|v| / 127.5)^2. Nodata -> NaN.
    Works on any array layout; the 64 unit-norm axes stay where they were."""
    v = arr.astype(np.float32)
    out = np.sign(v) * (np.abs(v) / 127.5) ** 2
    out[arr == AEF_NODATA] = np.nan
    return out


# --------------------------------------------------------------------------- TESSERA
def tessera_read_like(bbox: BBox, year: int, transform: Affine, crs: CRS, shape: tuple[int, int],
                      embeddings_dir: str | Path | None = None) -> np.ndarray:
    """Fetch TESSERA tiles covering `bbox` and resample them onto a target grid
    (transform, crs, shape=(H, W)) with nearest-neighbour. Returns float32 (128, H, W),
    NaN where TESSERA has no data. Use the AlphaEarth grid as target to compare pixels."""
    from geotessera import GeoTessera

    w, s, e, n = _check_bbox(bbox)
    cache = Path(embeddings_dir) if embeddings_dir else DATA_DIR / "cache" / "tessera"
    cache.mkdir(parents=True, exist_ok=True)
    gt = GeoTessera(embeddings_dir=str(cache))
    tiles = gt.registry.load_blocks_for_region(bounds=(w, s, e, n), year=year)
    # load_blocks_for_region pads with neighbours; keep tiles whose 0.1 deg box touches the bbox
    tiles = [t for t in tiles if t[1] + 0.05 > w and t[1] - 0.05 < e and t[2] + 0.05 > s and t[2] - 0.05 < n]
    H, W = shape
    out = np.full((128, H, W), np.nan, dtype=np.float32)
    for yr, tlon, tlat, arr, tcrs, ttr in gt.fetch_embeddings(tiles):
        src = np.moveaxis(arr, -1, 0)  # (128, h, w)
        valid = np.linalg.norm(src, axis=0) > 0
        src = np.where(valid[None], src, np.nan)
        piece = np.full_like(out, np.nan)
        reproject(src, piece, src_transform=ttr, src_crs=tcrs, dst_transform=transform, dst_crs=crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
        take = ~np.isnan(piece[0])
        out[:, take] = piece[:, take]
        log.info("tessera tile (%s, %s) %s -> pasted %d px", tlon, tlat, arr.shape, int(take.sum()))
    return out


# --------------------------------------------------------------------------- utilities
def pca_rgb(emb: np.ndarray, n_fit: int = 50_000, seed: int = 0, stretch: tuple[float, float] = (2, 98)) -> np.ndarray:
    """Turn a (D, H, W) embedding cube into an (H, W, 3) uint8 false-colour image from the
    first three principal components. NaN pixels come out black."""
    from sklearn.decomposition import PCA

    D, H, W = emb.shape
    flat = emb.reshape(D, -1).T  # (H*W, D)
    valid = ~np.isnan(flat).any(axis=1)
    rgb = np.zeros((H * W, 3), dtype=np.uint8)
    if valid.sum() < 3:
        return rgb.reshape(H, W, 3)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(valid)
    fit_idx = rng.choice(idx, size=min(n_fit, idx.size), replace=False)
    pca = PCA(n_components=3, random_state=seed).fit(flat[fit_idx])
    comps = pca.transform(flat[valid])
    lo, hi = np.percentile(comps, stretch, axis=0)
    comps = np.clip((comps - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    rgb[valid] = (comps * 255).astype(np.uint8)
    return rgb.reshape(H, W, 3)


def lonlat_extent(transform: Affine, crs: CRS, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """(west, east, south, north) in degrees, ready for matplotlib `extent=`."""
    H, W = shape
    left, top = transform.c, transform.f
    right, bottom = left + W * transform.a, top + H * transform.e
    w, s, e, n = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    return w, e, s, n


def to_lonlat_image(img: np.ndarray, transform: Affine, crs: CRS) -> tuple[np.ndarray, list[list[float]]]:
    """Reproject an (H, W, 3) uint8 image to EPSG:4326 for web-map overlays.
    Returns (image, [[south, west], [north, east]]) as folium expects."""
    from rasterio.warp import calculate_default_transform

    H, W, _ = img.shape
    left, top = transform.c, transform.f
    right, bottom = left + W * transform.a, top + H * transform.e
    dst_tr, dw, dh = calculate_default_transform(crs, "EPSG:4326", W, H, left, bottom, right, top)
    out = np.zeros((3, dh, dw), dtype=np.uint8)
    reproject(np.moveaxis(img, -1, 0), out, src_transform=transform, src_crs=crs,
              dst_transform=dst_tr, dst_crs="EPSG:4326", resampling=Resampling.nearest)
    w, e, s, n = lonlat_extent(dst_tr, CRS.from_epsg(4326), (dh, dw))
    return np.moveaxis(out, 0, -1), [[s, w], [n, e]]


def lonlat_to_rowcol(lons: Sequence[float], lats: Sequence[float], transform: Affine, crs: CRS) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised lon/lat -> (rows, cols) on a raster grid."""
    xs, ys = warp_transform("EPSG:4326", crs, list(lons), list(lats))
    inv = ~transform
    cols, rows = zip(*(inv * (x, y) for x, y in zip(xs, ys)))
    return np.floor(rows).astype(int), np.floor(cols).astype(int)


def sample(emb: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pick embedding vectors at pixel positions from a (D, H, W) cube -> (N, D)."""
    D, H, W = emb.shape
    ok = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    if not ok.all():
        raise IndexError(f"{(~ok).sum()} points fall outside the raster")
    return emb[:, rows, cols].T



def resample_like(arr: np.ndarray, src_transform: Affine, dst_transform: Affine, dst_shape: tuple[int, int],
                  crs: CRS, method: str = "nearest") -> np.ndarray:
    """Put a 2-D array (or a bool mask) from one grid onto another grid in the same CRS.
    method: 'nearest' | 'max' | 'average'. Bool input -> bool output (max = any)."""
    was_bool = arr.dtype == bool
    src = arr.astype(np.float32)
    dst = np.zeros(dst_shape, dtype=np.float32)
    res = {"nearest": Resampling.nearest, "max": Resampling.max, "average": Resampling.average}[method]
    reproject(src, dst, src_transform=src_transform, src_crs=crs, dst_transform=dst_transform, dst_crs=crs, resampling=res)
    return dst > 0 if was_bool else dst


def cosine_similarity_map(emb: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pixel of a (D, H, W) cube and a (D,) query vector."""
    q = query / (np.linalg.norm(query) + 1e-12)
    norms = np.linalg.norm(emb, axis=0)
    return np.einsum("dhw,d->hw", emb, q) / (norms + 1e-12)
