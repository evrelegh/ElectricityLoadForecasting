# Electricity Load Forecasting

A reproducible study of probabilistic day-ahead electricity load forecasting for the Belgian power system, using public Elia data.

## Research question

How much of Belgian quarter-hourly electricity demand can be predicted one day ahead from temporal structure and recent load information alone?

And once a point forecast has been made, how well can its uncertainty be quantified?

The project deliberately uses transparent statistical models rather than a large forecasting model zoo. The objective is not to win a retrospective forecasting contest, but to determine what information is genuinely predictive under an operational information boundary.

## Forecasting setup

Forecasts are made for every 15-minute period of the following Belgian civil day, using only information available by **18:00 Europe/Brussels on D−1**, matching the stated origin of Elia's published day-ahead forecast.

The modelling sequence is:

1. civil-time persistence benchmarks, including previous week D−7;
2. a Fourier/calendar model for recurring daily and weekly load structure;
3. a leakage-safe recent-level correction using forecast errors observed during the preceding 24 elapsed hours;
4. empirical P10–P90 uncertainty intervals based on recent historical forecast residuals for the same Belgian civil hour.

Daylight-saving transitions, holidays, missing observations and forecast availability are handled explicitly.

## Untouched confirmation

Model development used data before the final confirmation period.

The complete point and probabilistic specification was then **frozen in Git before any untouched 2025 confirmation result was scored**. January 2025 had already been encountered during development and was therefore excluded from untouched claims.

The frozen method was subsequently evaluated, unchanged, on **1 February–31 December 2025**.

### Point forecasts

| Forecast | MAE (MW) |
|---|---:|
| Previous week (D−7) | 438.9 |
| D−7 + identical recent-level correction | 423.0 |
| Fourier/calendar | 501.1 |
| **Fourier/calendar + recent level** | **315.4** |
| Elia day-ahead | 277.7 |

The frozen model improved MAE by **25.4%** relative to the information-matched D−7 benchmark receiving the same recent-level correction.

A seven-day moving-block bootstrap, preserving within-day and short-range between-day dependence, gave a 95% interval of **18.8% to 32.0%** for that improvement.

The result suggests that recurring load shape and current load level contain complementary predictive information. Recent-level information alone does not explain the gain.

Elia's operational forecast nevertheless remained better: the frozen model's MAE was **13.6% higher**, with a 95% block-bootstrap interval of **5.9% to 21.5%**.

## Probabilistic forecasts

The frozen empirical-residual P10–P90 intervals achieved:

| Forecast | Coverage | Mean width (MW) | Mean pinball loss |
|---|---:|---:|---:|
| Frozen empirical intervals | 0.765 | 941.9 | 74.19 |
| Elia day-ahead | 0.758 | 836.1 | 66.09 |
| Nominal | 0.800 | — | — |

The 95% block-bootstrap interval for frozen-model coverage was **0.725–0.800**. Its aggregate coverage was not distinguishable from Elia's: the paired coverage difference was +0.007 with interval **−0.033 to +0.042**.

But equivalent coverage was bought at a cost. Relative to Elia, the frozen intervals were:

- **12.7% wider** — 95% interval **+5.8% to +20.1%**;
- **12.3% worse in mean pinball loss** — 95% interval **+3.5% to +22.4%**.

The simple empirical-residual uncertainty model therefore did not match the efficiency of Elia's operational probabilistic forecast.

## What the experiment found

The Fourier/calendar model by itself was not competitive: regular periodic structure is not sufficient for good day-ahead forecasting.

Its value appeared when combined with a leakage-safe estimate of the current load level. A control experiment applying the identical level correction to weekly persistence showed that the improvement was not merely a recency effect.

The probabilistic experiment produced a useful negative result. Recent empirical errors conditioned only on civil hour provide a reasonable uncertainty band, but not an especially efficient one. In the untouched confirmation, autumn calibration was particularly poor.

The study therefore identifies both **where a simple transparent forecast obtains useful predictive information and where that approach reaches its limits**.

## Methodological safeguards

The project treats temporal integrity as part of the model, not as a cleanup step.

- Belgian civil time is modelled explicitly, including 92-, 96- and 100-slot days.
- Forecasts cannot use observations beyond their stated forecast origin.
- Missing observations are not silently imputed for scoring.
- Baselines and candidate models are compared on common samples.
- Positive and negative leakage controls test the information boundary.
- The final 2025 confirmation specification was frozen before its outcomes were opened.
- Post-confirmatory uncertainty analysis resamples whole civil days in seven-day moving blocks.
- Negative results are retained rather than tuned away.

## Repository

    notebooks/probabilistic_forecasting.ipynb   complete research narrative
    src/electricity_load_forecasting/           reusable forecasting and validation code
    tests/                                      automated tests
    figures/                                    generated diagnostic figures

The notebook contains the empirical argument and dataset-specific audits. Reusable behaviour is implemented in the package and covered by tests.

## Running the project

```bash
pip install -e ".[dev]"
pytest
jupyter lab notebooks/probabilistic_forecasting.ipynb
