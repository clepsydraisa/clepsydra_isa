## Testes de tendências em séries temporais

Usar package `pyMannKendall` (https://github.com/mmhs013/pyMannKendall) para analisar a tendência.
> pyMannKendall: a python package for non parametric Mann Kendall family of trend tests https://joss.theoj.org/papers/10.21105/joss.01556

Considerar (2) para testar se há tendência na série temporal com autocorrelação mas sem sazonalidade. Ver (10) que pode ser mais geral. As funções em  (12) e (13) são para estimar de forma robusta a tendência da série.
- (2) Hamed and Rao Modified MK Test (`hamed_rao_modification_test`): This mod-
ified MK test was proposed by Hamed & Rao (1998) to address serial autocorrelation
issues. They suggested a variance correction approach to improve trend analysis. Users
can consider first n significant lag by insert lag number in this function. By default, it
considered all significant lags. (https://github.com/clepsydraisa/clepsydra_isa/blob/main/material_apoio/time_series/modifiedAutocorrelated_MK_test_1997.pdf)
- (10) Correlated Seasonal MK Test (`correlated_seasonal_test`): This method was pro-
posed by Hipel & McLeod (1994), for when time series significantly correlate with the
preceding one or more months/seasons.
- (12) Theil-sen’s Slope Estimator (`sens_slope`): This method was proposed by Theil
(1950) and Sen (1968) to estimate the magnitude of the monotonic trend.
- (13) Seasonal sen’s Slope Estimator (`seasonal_sens_slope`): This method was proposed
by Hipel & McLeod (1994) to estimate the magnitude of the monotonic trend, when
data has seasonal effects.

Método iterativo para estimar tendência e correlação. Aparentemente não existe código Python.
> Xiaolan L. Wang and Val R. Swail (2001). Changes of Extreme Wave Heights in Northern Hemisphere Oceans and Related Atmospheric Circulation Regimes, Journal of Climate, (https://journals.ametsoc.org/view/journals/clim/14/10/1520-0442_2001_014_2204_coewhi_2.0.co_2.xml) -- Ver Anexo A

## Predição em séries temporais
Para forecasting em geral, ver livro (on-line) [Forecasting: Principles and Practice](https://otexts.com/fpp2/). Este livro usa o package *forecast* de R. Para Python uma possibilidade é usar *forecast* com https://pypi.org/project/rpy2/.  Do livro: *Exponential smoothing and ARIMA models are the two most widely used approaches to time series forecasting, and provide complementary approaches to the problem. While exponential smoothing models are based on a description of the trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data.*

Artigo seminal para TBATS:
> De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011), Forecasting time series with complex seasonal patterns using exponential smoothing, Journal of the American Statistical Association, 106(496), 1513-1527.

Implementação de Exponential smoothing em Python: ver [scripts/predicao](https://github.com/clepsydraisa/clepsydra_isa/tree/main/scripts/predicao)
