<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# How to use TBATS

The **TBATS** model is designed for time series forecasting with complex and potentially multiple seasonal periods. It is widely used when data exhibits non-integer and overlapping seasonality, and is supported in Python via the `tbats` package and, more recently, the `sktime` framework.

### Core TBATS Features:

- **Trigonometric seasonality** (Fourier terms), allowing for multiple/complex periods
- **Box-Cox transformation**, to stabilize variance
- **ARMA errors** for autocorrelated residuals
- **Trend and damping** components
- Automated component selection using information criteria[^11][^13][^15][^16]


### Python Implementation

#### Installation

```bash
pip install tbats
```

For sktime (if you plan to use sci-kit-learn compatible pipelines):

```bash
pip install sktime
```


#### Basic Usage with the tbats Package

```python
from tbats import TBATS
import numpy as np

# Example synthetic data
np.random.seed(2342)
t = np.arange(160)
y = 5 * np.sin(t * 2 * np.pi / 12) + np.random.normal(size=160)

# Specify seasonal periods: for monthly data with annual seasonality, set to [^12]
estimator = TBATS(seasonal_periods=[^12])

# Fit the model
fitted_model = estimator.fit(y)

# Forecast next 12 months
forecast = fitted_model.forecast(steps=12)

# Model summary
print(fitted_model.summary())
```

Key methods after fitting:

- `fitted_model.forecast(steps=N)` for forecasts
- `fitted_model.y_hat` for in-sample predictions
- `fitted_model.resid` for residuals
- `fitted_model.aic` for information criterion (model comparison)[^10][^13][^15]


#### Usage with sktime (recommended for integration with modern ML workflows)

```python
from sktime.datasets import load_airline
from sktime.forecasting.tbats import TBATS

y = load_airline()  # Example univariate monthly airline passenger data

forecaster = TBATS(sp=12)  # sp = seasonal_periods, e.g., [^12] for monthly data
forecaster.fit(y)
y_pred = forecaster.predict(fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # 12-step forecast
```

TBATS in `sktime` is easily parallelizable via the `n_jobs` argument for larger datasets or model selection[^11].

### Parameter Guidance

- **seasonal_periods / sp**: List of integer or float seasonal cycles (e.g., `[^12]` for monthly, `[7, 365.25]` for daily and yearly in daily data)
- **use_box_cox**, **use_trend**, **use_damped_trend**, **use_arma_errors**: Control optional model components, often left as `None` for auto-selection.
- **n_jobs**: Parallel jobs for faster fitting on multi-core machines[^11][^13].


### Performance Considerations

- TBATS models are computationally intensive, especially for long time series or multiple seasonalities, but parallelism (`n_jobs`, `multiprocessing`, or leveraging BLAS parallelism) can partially address this[^11][^13].
- For custom GPU acceleration, TBATS is not natively supported, but you may pre-process the data on GPU and, if required, try batching or parallel TBATS model runs using multiprocessing, especially since your workflow already includes advanced parallel and GPU strategies.


### Example: Forecasting with Multiple Seasonality

```python
from tbats import TBATS

# For daily data with weekly and yearly cycles:
estimator = TBATS(seasonal_periods=[7, 365.25])
fitted_model = estimator.fit(y_train)
forecast = fitted_model.forecast(steps=90)
```

This will pick up both weekly and annual structures[^13][^15].

### Summary Table

| Library | Key Function | Seasonality | Parallelism | ML Integration |
| :-- | :-- | :-- | :-- | :-- |
| tbats | TBATS, fit, forecast | Multiple (list) | Multi-processing | Manual |
| sktime | TBATS | Multiple (`sp` argument) | n_jobs parameter | Full (sklearn-like) |

### Limitations

- TBATS is best for univariate series.
- Training can be slow for long series or when using many seasonal periods.
- Not directly GPU-accelerated; all computations are on CPU[^13][^16].

For deeper integration with just-in-time compilation or GPU acceleration, or for multivariate workflows, you may need to implement custom model wrappers or distribute training manually since TBATS does not leverage Numba, BLAS, RAPIDS, or CUDA by default.

For practical and up-to-date code snippets and guide, the tbats PyPI page and GitHub repo provide concise, working examples[^13][^10].

<div style="text-align: center">⁂</div>

[^1]: https://www.youtube.com/watch?v=f6dInUl_9iI

[^2]: https://www.youtube.com/watch?v=eKiXtGzEjos

[^3]: https://www.youtube.com/watch?v=saxSR5KHy8M

[^4]: https://www.youtube.com/watch?v=Bl2RHfE4Qnc

[^5]: https://www.youtube.com/watch?v=vV12dGe_Fho

[^6]: https://www.youtube.com/watch?v=uD88ecvVFdI

[^7]: https://www.youtube.com/watch?v=7K3Arf-U39E

[^8]: https://www.youtube.com/watch?v=pKtyTLARndk

[^9]: https://www.linkedin.com/pulse/tbats-python-tutorial-examples-ikigailabs

[^10]: https://github.com/intive-DataScience/tbats

[^11]: https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.tbats.TBATS.html

[^12]: https://www.slideshare.net/slideshow/tf-group-projectpptx/255429306

[^13]: https://pypi.org/project/tbats/

[^14]: https://rdrr.io/cran/forecast/man/tbats.html

[^15]: https://www.programmersought.com/article/15724378214/

[^16]: https://campus.datacamp.com/courses/forecasting-in-r/advanced-methods?ex=8

[^17]: https://robjhyndman.com/hyndsight/simulatingtbats/

[^18]: https://rpubs.com/abotalebmostafa/708905

