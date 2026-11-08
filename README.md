# Winam Gulf Water Quality Assessment

This project analyses satellite-derived water-quality indicators and algal-bloom patterns in Winam Gulf, Lake Victoria. The notebook uses Google Earth Engine (Sentinel-2 and Landsat imagery) from 2018 through the last complete calendar month at run time, and produces monthly summaries, plots, a monitoring brief, and image exports. Separate, locally testable analyses live in `winam_gulf/`.

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
python -m pip install -r requirements.txt
```

The notebook no longer installs or upgrades packages each time it runs.

Install the extra API and ipyleaflet packages in the same environment when using the live map/API integration:

```powershell
python -m pip install -r requirements-app.txt
```

For running the complete automated test suite, install the development requirements:

```powershell
python -m pip install -r requirements-dev.txt
```

## Run the analysis

1. Open `winam_gulf_water_quality.ipynb` and select the environment where you installed the dependencies.
2. Set `EARTH_ENGINE_PROJECT` to a Google Cloud project enabled for Earth Engine before launching VS Code/Jupyter. In PowerShell, for example:

   ```powershell
   $env:EARTH_ENGINE_PROJECT = "your-cloud-project-id"
   code .
   ```

   Alternatively, set `EE_PROJECT_ID` directly in the first code cell. The notebook intentionally contains no default project ID.
3. Run the notebook from the beginning. Complete Earth Engine authentication when prompted.
4. Follow the per-year progress messages while monthly water-quality and rainfall summaries are fetched.
5. Review the analysis summary and the last 12 monthly records in the export cell.

The authenticated account must be allowed to use the selected Cloud project. If initialization returns `USER_PROJECT_DENIED`, ask the project administrator for the `serviceusage.services.use` permission (commonly through `roles/serviceusage.serviceUsageConsumer`) or select a project you can use.

## Live FastAPI and GEE maps

The notebook uses `geemap` for the original interactive maps. A separate local FastAPI service now exposes monthly Sentinel-2 summaries and Earth Engine tile URLs; the notebook consumes that API and renders its NDCI and Floating Algae Index (FAI) layers with `ipyleaflet`.

1. In PowerShell, set the same Earth Engine project and authenticate once if this machine has not already been authenticated:

   ```powershell
   $env:EARTH_ENGINE_PROJECT = "your-cloud-project-id"
   earthengine authenticate
   ```

   The API uses the authenticated user's persistent Earth Engine credentials/Application Default Credentials. Do not put credential files, tokens, or service-account keys in the repository.
2. From the repository root, start the API in a terminal:

   ```powershell
   python -m uvicorn winam_gulf.api:app --host 127.0.0.1 --port 8000
   ```

3. In another terminal, confirm initialization before opening the notebook:

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/health
   ```

   Continue when the response reports `"gee_initialized": true`. The interactive API reference is available at `http://127.0.0.1:8000/docs`.
4. Run the notebook through its final **Live Earth Engine API and ipyleaflet maps** cell. It fetches the monthly time series from the API, compares overlapping NDCI values with the notebook's own GEE series, and renders the latest complete month with available scenes. Set `WINAM_GULF_API_URL` if the server is at a different local URL.

The API and notebook use `COPERNICUS/S2_SR_HARMONIZED` and the same JRC-derived Gulf AOI and 40% scene cloud-cover filter. The API labels the SR method as Sen2Cor; this integration does **not** substitute ACOLITE/C2RCC or fix aquatic atmospheric correction. NDCI/FAI values are relative spectral indices. API map URLs are short-lived Earth Engine tile URLs; keep the API bound to localhost and do not publish those URLs as permanent resources. The hyacinth-candidate API layer requires explicitly supplied local NDVI/FAI thresholds and is not shown in the notebook until calibrated.

The API endpoints are `GET /health`, `GET /api/v1/timeseries?start=YYYY-MM-DD&end=YYYY-MM-DD`, and `GET /api/v1/maps/{ndci|ndvi|fai|hyacinth_candidate}?month=YYYY-MM-DD`. Monthly summary requests are limited to 120 complete months. Supplying both `ndvi_threshold` and `fai_threshold` query parameters adds a separately labeled NDCI series/map with candidate floating vegetation screened and an explicit candidate-mask layer. Thresholds have no defaults: calibrate them against locally labeled vegetation before using the mask. OpenAPI documentation is available at `/docs`.

The core time series continues to use Sentinel-2 observations. Expensive optional analyses are off by default (`RUN_DEGLINT_CORRECTION`, `RUN_INSITU_CALIBRATION`, and `RUN_SENSOR_CROSS_CALIBRATION` are all `False`). Enable them explicitly when required; scene-level Hedley correction and in-situ calibration can substantially increase runtime or downloads. The notebook prints the selected settings and records the deglint choice in the CSV.

The monthly composites and regional statistics are computed on Earth Engine and retrieved in yearly batches rather than making several blocking requests for every month. Rainfall is also summarized by year. Progress is printed before each batch so a remote Earth Engine request is visible while it runs.

The open-lake reference comparison uses its own Sentinel-2 collection filtered to the reference buffer. Keep the reference point over open water; the comparison reports an error with valid-month counts if there are no overlapping numeric observations.

## Bloom frequency and discharge points

The notebook calculates per-pixel bloom-proxy frequency from the monthly NDCI composites as the fraction of valid months with NDCI at or above `BLOOM_NDCI_THRESHOLD` (default `0.15`). This threshold is an exploratory elevated-risk boundary used by this notebook, not a universal or field-validated bloom threshold. A map is displayed in the notebook and a `winam_gulf_bloom_frequency` GeoTIFF export is started in Earth Engine Tasks.

The starter file `data/discharge_points_user_supplied.csv` contains candidate facilities and river mouths with coordinates supplied for this project. These locations have not been independently verified and are not an official register of active outfalls. The notebook filters to points within 10 km of the Winam Gulf AOI; Kampala, Jinja, and Mwanza points remain in the CSV for regional context but are excluded from the Gulf overlay and distance analysis. Before interpreting the output, verify coordinates, active status, discharge type, and permission to use the locations with Kenya's [National Environment Management Authority (NEMA)](https://nema.go.ke/), [Water Resources Authority (WRA)](https://wra.go.ke/), or the relevant utility. If using another point dataset, supply `name,lon,lat` in decimal-degree WGS84 CSV format or use a point GeoJSON, GeoPackage, or shapefile with a declared CRS, and set `DISCHARGE_DATA_PATH` in notebook Section 9. When local points are loaded, the notebook overlays them and runs distance-bin analysis, saving `winam_gulf_discharge_distance_decay.csv` and `distance_decay.png`. Treat any resulting association as exploratory, not causal.

## Optional calibration analyses

Three longer analyses are off by default so the core monitoring run avoids large downloads and many remote requests. To include one, change its setting in the first code cell before running the notebook:

- `RUN_INSITU_CALIBRATION = True` downloads the approximately 152 MB in-situ workbook if it is not already present and matches all observations in one batched Earth Engine request.
- `RUN_SENSOR_CROSS_CALIBRATION = True` compares up to 12 date-spread Sentinel-2/Landsat scene pairs for the TSS and CDOM proxies.

Each disabled analysis prints a message instead of silently appearing to run. In-situ calibration is an optional validation step; without it, the reported NDCI values remain satellite-derived proxies rather than calibrated chlorophyll-a concentrations.

### Regional Lake Victoria calibration dataset

The optional regional calibration uses the Deirmendjian et al. workbook from the public [Zenodo record 4247352](https://zenodo.org/records/4247352). When `RUN_INSITU_CALIBRATION = True`, the notebook downloads the workbook automatically to `deirmendjian_lake_victoria_2020.xlsx` if it is not already present. The workbook is excluded from Git by `.gitignore` because it is a large external dataset; each user can retrieve it from Zenodo as needed. This is a Ugandan Lake Victoria dataset, not local Winam Gulf validation. The workbook contains 79 surface observations (not 219 independent surface visits), and its recorded dates should be reviewed against the campaign description before interpreting calibration results.

## Outputs

The notebook writes local outputs to the current working directory:

- `winam_gulf_water_quality_timeseries.csv` — monthly proxy values, scene counts, confidence flags, and whether Hedley sun-glint correction was applied
- `winam_gulf_vs_reference_timeseries.csv` — comparison with the reference site, when that section runs
- `winam_gulf_timeseries.png` and `winam_gulf_mean_vs_max.png` — time-series plots
- `winam_gulf_monthly_report.pdf` — a one-page monitoring brief
- `winam_gulf_dashboard.html`, `winam_gulf_dashboard.png`, and `winam_gulf_summary.json` — standalone visual and machine-readable results
- `bloom_forecast.png` — observed NDCI with the seasonal historical reference, when data requirements are met
- `winam_gulf_bloom_frequency` — Earth Engine Drive export of the per-pixel bloom-proxy frequency raster
- `winam_gulf_discharge_distance_decay.csv` and `distance_decay.png` — generated only when verified discharge-point data is supplied

The final notebook cell starts asynchronous Google Drive exports for the latest-month index image and the bloom-frequency raster in the `WinamGulf_WQ_Outputs` folder. Track completion in the Earth Engine Tasks page; the notebook does not wait for these exports to finish.

The notebook also builds a standalone `winam_gulf_dashboard.html`, an overview chart at `winam_gulf_dashboard.png`, and a machine-readable `winam_gulf_summary.json`. The dashboard includes observed proxy time series, monthly scene counts, recent records, the trend-test summary, and an optional seasonal reference. The seasonal reference uses same-calendar-month historical medians and a descriptive percentile range; it is not a validated prediction. The trend test uses actual elapsed time for Sen's slope and reports that its p-value is not adjusted for serial correlation.

## Separate science extensions

The notebook stays focused on Earth Engine extraction. Analysis utilities, input templates, and protocols for correction-product comparisons, floating-vegetation screening, field validation, driver association, flow-path distance, and held-out forecast skill are in [docs/science_extensions.md](docs/science_extensions.md). They require actual corrected scenes and verified field/driver data; no sample values are presented as Winam Gulf results.

Run the offline regression tests with:

```powershell
python -m pytest
```

Examples of separate analyses:

```powershell
python -m winam_gulf.cli corrections --input path\to\correction_monthly.csv --output-dir outputs\corrections
python -m winam_gulf.cli forecast --input winam_gulf_water_quality_timeseries.csv --output-dir outputs\forecast
```

## Interpretation notes

- Chlorophyll-a (NDCI), TSS, and CDOM are satellite-derived proxies; do not report them as measured concentrations without calibration.
- Cloud cover, missing observations, and low-scene-count months affect confidence. The CSV includes scene counts and a low-confidence flag.
- Rainfall correlation does not establish causation, and the catchment region is approximate.
- The bloom-frequency raster is a threshold-exceedance proxy; discharge-point distance analysis requires verified outlet data. Neither satellite proxy nor spatial association alone proves a pollution source or causal effect.
