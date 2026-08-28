# Electricity Load Forecasting

A reproducible study of Belgian day-ahead electricity load forecasting on public
Elia system data.

Electricity demand is strongly periodic, but periodicity is not the same as
predictability. This project asks how much of Belgian quarter-hourly load can be
forecast one day ahead from temporal structure alone, and whether the resulting
uncertainty estimates are statistically calibrated.

## Approach

Every result is required to survive an independent check before it is used.
Point accuracy and probabilistic calibration are scored on their own samples,
assumptions are stated where they are made, and a failing contract raises rather
than being repaired silently. A negative result is reported as a result.

## Current stage

**1. Audit of the operational forecast.** Before modelling anything, Elia's own
published day-ahead forecast for 2023 is audited: bias, MAE and RMSE; empirical
coverage of the P10-P90 band; interval width; pinball loss; and calibration by
hour, month, day type and load level. Belgian civil time, the daylight-saving
transitions and public holidays are handled explicitly.

**2. Spectral analysis.** The realised load series is decomposed by
periodogram. Peaks are located by prominence and only then read as periods, so
the daily and weekly cycles are found rather than assumed. The result is
cross-checked against an independent gap fill and against an unwindowed
transform.

## Repository

    notebooks/probabilistic_forecasting.ipynb   research narrative and data contracts
    src/electricity_load_forecasting/           tested reusable core
    tests/                                      pytest suite
    figures/                                    generated figures

The notebook carries the argument and the dataset-specific contracts. General
behaviour — civil-day length across DST, the holiday calendar, scoring rules,
spectral estimation, the temporal-integrity guard — lives in the package and is
tested there.

## Running it

    pip install -e ".[dev]"
    pytest
    jupyter lab notebooks/probabilistic_forecasting.ipynb

The notebook pulls Elia `ods001` on first run and caches it under `cache/`,
which is deliberately not committed: the data is reproducible from the source.

## Planned

Transparent day-ahead benchmarks (previous day, previous week), a Fourier and
calendar model, rolling-origin backtesting, and probabilistic forecasts scored
for calibration and sharpness against the Elia reference.

## Data source

Public Elia Open Data, dataset `ods001` (measured and forecast total load,
Belgian control zone).
