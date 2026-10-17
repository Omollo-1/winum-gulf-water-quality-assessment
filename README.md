# Winam Gulf Water Quality Assessment

This notebook analyses satellite-derived water-quality indicators and algal-bloom patterns in Winam Gulf, Lake Victoria, from 2018 through 2024. It uses Google Earth Engine (Sentinel-2 and Landsat imagery) and produces monthly summaries, plots, a monitoring brief, and a latest-month image export.

## Requirements

- Python 3.10 or later
- A Google Earth Engine account with access to the Earth Engine API
- A Google Cloud project enabled for Earth Engine
- Jupyter support in VS Code or another Jupyter notebook client

Create and activate a virtual environment in PowerShell, then install the notebook dependencies once:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install earthengine-api geemap geopandas rasterio scikit-learn scipy statsmodels pandas numpy matplotlib requests openpyxl tenacity
```

The notebook no longer installs or upgrades packages each time it runs.

## Run the analysis

1. Open `winam_gulf_water_quality.ipynb` and select the environment where you installed the dependencies.
2. In the first code cell, change the Earth Engine project ID in `ee.Initialize(project=...)` to your project.
3. Run the notebook from the beginning. Complete Earth Engine authentication when prompted.
4. Follow the per-year progress messages while monthly water-quality and rainfall summaries are fetched.
5. Review the final summary and the last 12 monthly records in the last cell.

The authenticated account must be allowed to use the selected Cloud project. If initialization returns `USER_PROJECT_DENIED`, ask the project administrator for the `serviceusage.services.use` permission (commonly through `roles/serviceusage.serviceUsageConsumer`) or select a project you can use.

The core time series continues to use Sentinel-2 observations. For a faster default run, scene-level Hedley sun-glint correction is off (`RUN_DEGLINT_CORRECTION = False`): it samples and fits each scene and can substantially increase Earth Engine work. Set it to `True` when corrected reflectance is required; report that choice when interpreting NDCI results. The notebook prints which preprocessing path it uses.

The monthly composites and regional statistics are computed on Earth Engine and retrieved in yearly batches rather than making several blocking requests for every month. Rainfall is also summarized by year. Progress is printed before each batch so a remote Earth Engine request is visible while it runs.

The open-lake reference comparison uses its own Sentinel-2 collection filtered to the reference buffer. Keep the reference point over open water; the comparison reports an error with valid-month counts if there are no overlapping numeric observations.

## Bloom frequency and discharge points

The notebook calculates per-pixel bloom-proxy frequency from the monthly NDCI composites as the fraction of valid months with NDCI at or above `BLOOM_NDCI_THRESHOLD` (default `0.15`). This threshold is an exploratory elevated-risk boundary used by this notebook, not a universal or field-validated bloom threshold. A map is displayed in the notebook and a `winam_gulf_bloom_frequency` GeoTIFF export is started in Earth Engine Tasks.

Discharge sites must come from verified records, not guessed coordinates. Request permitted effluent-outfall/wastewater-facility coordinates from Kenya's [National Environment Management Authority (NEMA)](https://nema.go.ke/), [Water Resources Authority (WRA)](https://wra.go.ke/), or the relevant county water/sewer utility. Ask for site/facility name or permit ID, longitude/latitude, receiving water, coordinate reference system, date/status, and permission to use the locations. Save as `name,lon,lat` in a CSV (decimal degrees, WGS84), or use a point GeoJSON, GeoPackage, or shapefile with a declared CRS. Set `DISCHARGE_DATA_PATH` in notebook Section 9 to the file path. When loaded, points are overlaid on the map and the notebook runs distance-bin analysis, saving `winam_gulf_discharge_distance_decay.csv` and `distance_decay.png`. Without verified point data, the distance-decay step is intentionally skipped.

## Optional calibration analyses

Two longer analyses are off by default so the core monitoring run avoids large downloads and many remote requests. To include one, change its setting in the first code cell before running the notebook:

- `RUN_INSITU_CALIBRATION = True` downloads the approximately 152 MB in-situ workbook if it is not already present and matches all observations in one batched Earth Engine request.
- `RUN_SENSOR_CROSS_CALIBRATION = True` compares up to 12 date-spread Sentinel-2/Landsat scene pairs for the TSS and CDOM proxies.

Each disabled analysis prints a message instead of silently appearing to run. In-situ calibration is an optional validation step; without it, the reported NDCI values remain satellite-derived proxies rather than calibrated chlorophyll-a concentrations.

## Outputs

The notebook writes local outputs to the current working directory:

- `winam_gulf_water_quality_timeseries.csv` — monthly proxy values, scene counts, confidence flags, and whether Hedley sun-glint correction was applied
- `winam_gulf_vs_reference_timeseries.csv` — comparison with the reference site, when that section runs
- `winam_gulf_timeseries.png` and `winam_gulf_mean_vs_max.png` — time-series plots
- `winam_gulf_monthly_report.pdf` — a one-page monitoring brief
- `winam_gulf_bloom_frequency` — Earth Engine Drive export of the per-pixel bloom-proxy frequency raster
- `winam_gulf_discharge_distance_decay.csv` and `distance_decay.png` — generated only when verified discharge-point data is supplied

The final notebook cell starts asynchronous Google Drive exports for the latest-month index image and the bloom-frequency raster in the `WinamGulf_WQ_Outputs` folder. Track completion in the Earth Engine Tasks page; the notebook does not wait for these exports to finish.

## Interpretation notes

- Chlorophyll-a (NDCI), TSS, and CDOM are satellite-derived proxies; do not report them as measured concentrations without calibration.
- Cloud cover, missing observations, and low-scene-count months affect confidence. The CSV includes scene counts and a low-confidence flag.
- Rainfall correlation does not establish causation, and the catchment region is approximate.
- The bloom-frequency raster is a threshold-exceedance proxy; discharge-point distance analysis requires verified outlet data. Neither satellite proxy nor spatial association alone proves a pollution source or causal effect.
