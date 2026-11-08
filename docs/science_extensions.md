# Science extension workflows

The notebook remains responsible for Earth Engine authentication, scene discovery,
and the existing Gulf-wide time series. Offline comparisons and analyses live in
`winam_gulf/` so they can be tested independently and run against documented CSV
inputs. These tools do not create missing satellite, field, or river-flow data.

The local API in `winam_gulf/api.py` and service in `winam_gulf/gee_service.py`
provide a live GEE path: initialize from `EARTH_ENGINE_PROJECT` and existing
Earth Engine credentials, query Sentinel-2 SR monthly indices, and issue
temporary map tile URLs. The notebook's last integration cell reads those
endpoints and renders them with `ipyleaflet` using `winam_gulf/map_viewer.py`.
Start the API and validate `/health` before running that cell, as described in
the repository README. This API is a separate access path for live GEE data; the
offline CSV analyses below are not silently substituted with synthetic data.

## Atmospheric-correction comparison

Earth Engine's Sentinel-2 SR Harmonized collection is a Sen2Cor surface-reflectance
product. It is retained as the comparison baseline, not treated as an independently
validated aquatic correction. ACOLITE or C2RCC processing must be run separately
with the relevant software and inputs; this repository does not reprocess scenes or
claim that a correction is now fixed.

1. Select the same clear Sentinel-2 acquisitions across years and seasons. Keep
   the original scene IDs, UTC acquisition dates, processing versions, settings,
   QA decisions, and failures in a run log.
2. Process the selected acquisitions with ACOLITE and/or C2RCC. Use compatible
   atmospherically corrected reflectance products; do not compare reflectance
   values with incompatible units or scale factors.
3. Apply a consistent cloud, shoreline, and open-water sampling mask to all
   methods. Extract the same NDCI formula, spatial statistic, Gulf region, and
   calendar-month aggregation for each correction method.
4. Put the resulting monthly series in
   `templates/correction_monthly_template.csv`. Use method names `gee_s2_sr`,
   `acolite`, and/or `c2rcc`; each row is one method-month median. Include only
   months measured under the same sampling protocol.
5. Run:

   ```powershell
   python -m winam_gulf.cli corrections --input path\to\correction_monthly.csv --output-dir outputs\corrections
   ```

   The outputs report paired-month bias, MAE, RMSE, agreement correlation,
   Kendall trend direction, and Sen-slope differences, with an overlay plot.
   The report is conditional on the supplied data; it does not assert that trends
   agree. A reviewer-ready robustness check should disclose the matched dates,
   number of paired months, processing settings, and spatial mask.

## Floating-vegetation screening and Sentinel-3

`winam_gulf.spectral` provides NDVI, Floating Algae Index (FAI), a normalized
red-edge/red index, three-band line height, and an OLCI 681/709/754 nm index pair.
FAI requires reflectance in a red, NIR, and SWIR band; Sentinel-2's 20 m SWIR
measurement must be aligned with the selected NIR/red grid before calculation.
The functions return continuous indices. `hyacinth_candidate_mask` requires
explicit locally calibrated NDVI and FAI thresholds and intentionally has no
guessed defaults.

The GEE API also accepts both calibrated thresholds on the monthly-summary or
NDCI-map request. It returns an additional `ndci_excluding_candidate_vegetation`
summary and a separately labeled screened NDCI tile. The default unmasked NDCI
remains available for side-by-side comparison. When both thresholds are absent,
no mats are removed; when only one is supplied, the API rejects the request.

For defensible separation, label representative floating-hyacinth mats, open
water, and algal-bloom pixels across seasons; tune thresholds on training
locations/dates and report errors on separate dates or locations. Export the
candidate mask as a provisional layer, retain the original NDCI layer, and
compare NDCI trends with and without candidate mats. FAI/NDVI alone cannot
guarantee that floating vegetation is water hyacinth.

Sentinel-3 OLCI has different spatial resolution, spectral response, and revisit
characteristics from Sentinel-2. The OLCI index functions are spectral
calculations only; they do not ingest or harmonize satellite scenes. Match
overlapping dates/regions, use sensor-specific masks, and treat OLCI as a
separate validation/supplementary series until a cross-sensor calibration is
supported by paired observations.

## Field validation with Beach Management Units

Use `templates/field_validation_template.csv` for matched satellite/field
observations. The helper evaluates one field variable, unit, sensor-processing
method, and proxy at a time. With three or more unique sites it holds out sites
one at a time; with fewer sites it uses leave-one-observation-out and flags this
weaker validation design.

Before collecting data, agree with participating BMUs and the relevant
authorities on consent, safety, site access, and data sharing. Record UTC time,
WGS84 coordinates, instrument and calibration checks, replicate readings,
weather/sky conditions, depth, and a consistent sampling protocol. Secchi depth
and turbidity are useful validation targets for clarity/suspended matter but
are not chlorophyll-a measurements. A chlorophyll calibration requires measured
chlorophyll-a. Keep raw readings, units, and QA flags; do not infer missing
field values from satellite indices.

Run:

```powershell
python -m winam_gulf.cli field-validation --input path\to\matched_field_samples.csv --output-dir outputs\field
```

The outputs contain leave-out predictions, MAE, RMSE, out-of-fold R-squared,
and a full-sample calibration equation. Thirty to fifty points are a useful
starting goal, not a guarantee of geographic or seasonal representativeness.

## Regional Lake Victoria chlorophyll-a calibration

The notebook can use Deirmendjian et al.'s [Zenodo dataset 4247352](https://doi.org/10.5281/zenodo.4247352)
and the associated 2021 *Freshwater Biology* paper to fit a regional NDCI-to-
chlorophyll-a equation. The field sites are in Ugandan waters, so this is not
Winam Gulf validation and its coefficients are not applied to the Winam series.

The workbook's `Discrete-Profiles` sheet has 219 depth-profile rows, including
79 rows labeled `Surface`. The loader uses those surface observations only and
converts µg/L to the numerically equivalent mg/m³. It reconciles visit dates
from `Profile_exo_probe` using mission and coordinates; unresolved dates remain
visible and are not matched to imagery. In the current Zenodo workbook, 73 of
the 79 surface rows resolve to a unique probe-sheet date.

When enabled, the notebook samples Sentinel-2 SR Harmonized (Sen2Cor) NDCI at
the field locations, using a clear-pixel mask, an NDWI open-water mask, and a
20 m spatial summary. For each visit it selects the nearest valid scene within
±3 days. Leave-one-site-out calibration requires at least three matched site
IDs. With fewer matches, the notebook reports insufficient data and does not
fit or report calibration coefficients. Otherwise it fits a simple linear
equation and reports leave-one-site-out MAE, RMSE, R², and predictions. This is
regional model evidence only; it is not a local Gulf calibration or an
atmospheric-correction comparison. The notebook writes matched/unmatched
observation tables and metrics JSON to `outputs/regional_chla_calibration/`;
predictions and a plot are produced only when the minimum site requirement is
met.

**Date caveat:** the Zenodo description lists March/June/October campaigns, but
some dates stored in the workbook's probe sheet fall outside those months. The
loader preserves the workbook's date values without guessing a correction, and
the notebook displays these records before matching. Verify the source-date
interpretation before using or publishing the resulting coefficients.

## Environmental-driver association

Prepare one row per month in `templates/monthly_drivers_template.csv`. Use
documented sources, units, catchment definitions, and versions for CHIRPS
rainfall, ERA5 wind speed, and Nyando/Sondu-Miriu inflows. The existing GEE
rainfall summary can supply the rainfall series; river gauge/inflow values
require verified observed or modelled records and coverage metadata.

Run:

```powershell
python -m winam_gulf.cli drivers --input path\to\monthly_drivers.csv --output-dir outputs\drivers
```

The analysis uses a chronological train/holdout split, a Random Forest,
train-fitted missing-value imputation, a seasonal climatology baseline, and
permutation importance on the later holdout. Same-month and up-to-two-month
lags are included. Optional SHAP output can be requested with `--shap` after
installing `shap`. Predictive feature importance is association, not causal
attribution: omitted nutrients, co-varying drivers, measurement error, and
hydrodynamics remain potential confounders.

## Flow-path distance

Supply verified `distance_along_flow_m` values derived from a documented
water-network/hydrodynamic route in `templates/flowpath_observations_template.csv`.
Do not substitute Euclidean distance. Run:

```powershell
python -m winam_gulf.cli flowpath --input path\to\flowpath_observations.csv --output-dir outputs\flowpath
```

The output summarizes linear and rank associations; spatial association alone
does not identify a source or establish transport causality.

## Forecast skill

The default test trains through December 2022 and reserves January 2023 through
December 2024. The climatology baseline uses training-only same-calendar-month
medians. The candidate is a trend-plus-month-effects linear model fit only to
the training period. Both are scored on the same valid holdout months using
MAE, RMSE, bias, and R-squared.

```powershell
python -m winam_gulf.cli forecast --input winam_gulf_water_quality_timeseries.csv --output-dir outputs\forecast
```

Treat the candidate as unsupported unless it beats climatology on predeclared
holdout metrics. Once results are inspected, do not retune on 2023–2024 and
continue to call it an untouched test set.

## Operational alerts and publication

The repository can produce local dashboard/report outputs, but it does not send
SMS or WhatsApp messages. Operational alerts require field-validated thresholds,
approved wording, an accountable monitoring operator, BMU consent/contact
management, message costs, and a documented false-alarm/escalation process.
Do not publish an unvalidated satellite proxy as a health/safety alert.

Before archiving a GitHub release or requesting a Zenodo DOI, verify authorship,
license, source/data permissions, reproducible environment, a clean analysis
run, data provenance, and limitations. No DOI or publication-ready validation
claim is created by these extension scripts.
