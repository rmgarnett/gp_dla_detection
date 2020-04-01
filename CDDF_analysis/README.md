# Manipulate the catalogue

- To reproduce the CDDF, dN/dX, OmegaDLA plots in Bird (2017), use `calc_cddf.py`
- To manipulate the MATLAB catalogue without `sample_log_likelihoods_dla`, use `qsoloader.py`

## Basic usage of QSOLoader

This is how to instantiate this little class:

```python
from CDDF_analysis.qso_loader import QSOLoader

# in python
qsos = QSOLoader(
    preloaded_file="preloaded_qsos.mat", catalogue_file="catalog.mat",
    learned_file="learned_qso_model_dr9q_minus_concordance.mat", processed_file="processed_qsos_multi_dr12q.mat",
    dla_concordance="dla_catalog", los_concordance="los_catalog",snrs_file="snrs_qsos_multi_dr12q.mat",
    sub_dla=True)
```

- `preloaded_qsos.mat` and `catalog.mat` are the same as Garnett (2017).
- `learned_file`, `processed_file`, and `snrs_file` are re-processed with the multi-DLA pipeline.
- For concordance catalogues, they are in the path `data/dla_catalogs/dr9q_concordance/processed/dla_catalog` and `data/dla_catalogs/dr9q_concordance/processed/los_catalog` if you have run this before:

```bash
# in shell
cd data/scripts
./download_catalogs.sh
```

- If you are using a model considering sub-DLAs as a alternative model, use argument `sub_dla=True`.

The most useful feature is to plot a given spectrum with the GP mean prior:

```python
# the index of the catalogue
nspec = 1

qsos.plot_this_mu(nspec)
```

The above line will plot the GP prior multiplied with the maximum a posteriori (MAP) DLAs on top of the flux of the spectrum in the rest-frame with default settings.

Alternatively, you can change the level of mean-flux by `num_forest_lines` argument.
Also, you can add the Hydrogen lines for DLAs by changing `num_voigt_lines`.
If you are interested to know the predictions from Parks (2018), you can set `Parks=True` and specify the path to Parks' JSON catalogue `dla_parks="predictions_DR12.json"`.

For example, for suppressing mean-flux with Lyman-series, three Hydrogen lines for DLAs, and plotting Parks' predictions:

```python
qsos.plot_this_mu(nspec, suppressed=True,
    num_voigt_lines=3, num_forest_lines=31,
    Parks=True, dla_parks="predictions_DR12.json"))
```

The plotting method is based on `matplotlib`.
So remember to setup your own `matplotlib` environment.

- `qsos.plot_raw_spectrum(nspec)`

Sometime it will be helpful to retrieve the raw spectrum.
So here we provide a method to plot the raw spectrum.
If you don't have the raw spectrum in the right path,
it will automatically download the spectra and plot it.

However, it requires `astropy.io.fits` to read the `fits` file.
You may try `qsos.retrieve_raw_spec` function to directly download the data for single spectrum
and plot it by yourselves
if you don't have an installed `astropy`.

- `qsos.all_log_nhis` and `qsos.all_z_dlas` : you can get the MAP values from these two arrays.

If you are interested to know how to plot `ROC` curve of CDDF for Parks and Noterdaeme, please refer to `make_multi_dla_plots.py`.

If you are interested to know how to plot CDDF for the multi-DLA catalogue, please refer to `calc_cddf.py`.
