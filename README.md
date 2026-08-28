\# Electricity Load Forecasting



A reproducible study of Belgian day-ahead electricity load forecasting using public Elia system data.



The first stage of the project audits Elia's operational day-ahead forecasts for 2023:



\- realised total load versus day-ahead point forecasts;

\- P10-P90 uncertainty intervals;

\- bias, MAE and RMSE;

\- empirical interval coverage and pinball loss;

\- calibration by hour, month, day type and load level;

\- explicit handling of Belgian civil time, daylight-saving transitions and public holidays.



The analysis is designed as a reproducible research workflow. Data acquisition, integrity checks and modelling assumptions are made explicit in the notebook.



Planned next steps include transparent benchmark models, spectral analysis, rolling-origin validation and probabilistic forecast calibration.



\## Notebook



`notebooks/probabilistic\_forecasting.ipynb`



\## Data source



Public Elia Open Data, dataset `ods001`.



\## Status



Work in progress.

