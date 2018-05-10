DLA detection pipleine for BOSS quasar spectra
==============================================

This code repository contains code to completely reproduce the DLA
catalog reported in

> R Garnett, S Ho, S Bird, and J Schnedier. Detecting Damped Lyman-α
> Absorbers with Gaussian Processes. [arXiv:1605.04460
> [astro-ph.CO]](https://arxiv.org/abs/1605.04460),

including all intermediate data products including the Gaussian
process null model described therein. The provided parameters should
exactly reproduce the catalog in that work; however, you may feel free
to modify these choices as you see fit.

The pipeline has multiple stages, outlined and documented below.

Loading catalogs and downloading spectra
----------------------------------------

The first step of the process is to load the DR12Q quasar catalog and
available DLA catalogs, extract some basic data such as redshift,
coordinates, etc., and apply some basic filtering to the spectra:

* spectra with z < 2.15 are filtered
* spectra identified in a visual survey to have broad absorption line
  (BAL) features are filtered

Relevant parameters in `set_parameters` that can be tweaked if desired:

    % preprocessing parameters
    z_qso_cut      = 2.15;                        % filter out QSOs with z less than this threshold

This process proceeds in three steps, alternating between the shell
and MATLAB.

First we download the raw catalog data:

    # in shell
	cd data/scripts
	./download_catalogs.sh

Then we load these catalogs into MATLAB:

	% in MATLAB
    set_parameters;
	build_catalogs;

The `build_catalogs` script will produce a file called `file_list` in
the `data/dr12q/spectra` directory containing relative paths to
yet-unfiltered SDSS spectra for download. The `file_list` output is
available in this repository in the same location. The next step is to
download the coadded "speclite" SDSS spectra for these observations
(warning: total download is 35 GB). The `download_spectra.sh` script
requires `wget` to be available. On OS X systems, this may be
installed easily with [Homebrew](http://brew.sh/index.html).

	# in shell
	cd data/scripts
	./download_spectra.sh

`download_spectra.sh` will download the observational data for the yet
unfiltered lines of sight to the `data/dr12q/spectra` directory.

Loading and preprocessing spectra
---------------------------------

Now we load these data, continue applying filters, and do some basic
preprocessing. The additional filters are:

* spectra that have no nonmasked pixels in the range [1310, 1325]
  Angstroms (QSO restframe) are filtered, as they cannot be normalized
* spectra with fewer than 200 nonmasked pixels in the range [911,
  1217] Angstroms (QSO restframe) are filtered.

The preprocessing steps are to:

* truncate spectra to only contain pixels in the range [911, 1217]
  Angstroms QSO rest
* normalize flux and noise variance by dividing by the median flux in
  the range [1310, 1325] Angstroms QSO rest

Relevant parameters in `set_parameters` that can be tweaked if
desired:

    % preprocessing parameters
    min_num_pixels = 200;                         % minimum number of non-masked pixels

    % normalization parameters
	normalization_min_lambda = 1310;              % range of rest wavelengths to use   Å
	normalization_max_lambda = 1325;              %   for flux normalization

    % file loading parameters
	loading_min_lambda = 910;                     % range of rest wavelengths to load  Å
	loading_max_lambda = 1217;

When ready, the MATLAB code to preload the spectra is:

    set_parameters;
	release = 'dr12q';

	file_loader = @(plate, mjd, fiber_id) ...
    (read_spec(sprintf('%s/%i/spec-%i-%i-%04i.fits', ...
        spectra_directory(release),                  ...
        plate,                                       ...
        plate,                                       ...
        mjd,                                         ...
        fiber_id)));

	preload_qsos;

The result will be a completed catalog data file,
`data/[release]/processed/catalog.mat`, with complete filtering
information and a file containing preloaded and preprocessed data for
the 162861 nonfiltered spectra,
`data/[release]/processed/preloaded_qsos.mat`.

Building GP models for quasar spectra
-------------------------------------

Now we build our models, including our Gaussian process null model for
quasar emission spectra and our model for spectra containing DLAs.

To build the null model for quasar emission spectra, we need to
indicate a set of spectra to use for training, which should be
nominally DLA-free. Here we select all spectra:

* in DR9
* not removed by our filtering steps during loading
* in the DR9 Lyman-alpha forest catalog, and
* not in the DR9 Lyman-alpha DLA concordance catalog

These particular choices may be accomplished with:

    training_release  = 'dr12q';
    dla_catalog_name = 'dr9q_concordance';
    train_ind = ...
        [' catalog.in_dr9                     & ' ...
         '(catalog.filter_flags == 0)         & ' ...
         ' catalog.los_inds(dla_catalog_name) & ' ...
         '~catalog.dla_inds(dla_catalog_name)'];

After specifying the spectra to use in `training_release` and
`train_ind`, we call `learn_qso_model` to learn the model.
To learn the model, you need the matlab toolbox minFunc from:

`https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html`

You should cd to the directory where you extracted this to and run:

    addpath(genpath(pwd));
    mexAll;

to initialise the addon the first time you use it.

Relevant parameters in `set_parameters` that can be tweaked if
desired:

    % null model parameters
    min_lambda         =  911.75;                 % range of rest wavelengths to       Å
    max_lambda         = 1215.75;                 %   model
    dlambda            =    0.25;                 % separation of wavelength grid      Å
    k                  = 20;                      % rank of non-diagonal contribution
    max_noise_variance = 1^2;                     % maximum pixel noise allowed during model training

    % optimization parameters
	initial_c     = 0.1;                          % initial guess for c
	initial_tau_0 = 0.0023;                       % initial guess for τ₀
	initial_beta  = 3.65;                         % initial guess for β
	minFunc_options =               ...           % optimization options for model fittin        struct('MaxIter',     2000, ...
               'MaxFunEvals', 4000);

When ready, the MATLAB code to learn the null quasar emission model
is:

    training_set_name = 'dr9q_minus_concordance';
    learn_qso_model;

The learned qso model is stored in
`data/[training_release]/processed/learned_qso_model_[training_set_name].mat`.

We also need to specify a set of DLA parameter samples for the DLA
model. This is handled by the `generate_dla_samples` script.

Relevant parameters in `set_parameters` that can be tweaked if
desired:

    % DLA model parameters: parameter samples
	num_dla_samples     = 10000;                  % number of parameter samples
	alpha               = 0.9;                    % weight of KDE component in mixture
	uniform_min_log_nhi = 20.0;                   % range of column density samples    [cm⁻²]
	uniform_max_log_nhi = 23.0;                   % from uniform distribution
	fit_min_log_nhi     = 20.0;                   % range of column density samples    [cm⁻²]
	fit_max_log_nhi     = 22.0;                   % from fit to log PDF

When ready, the MATLAB code to generate the DLA model parameter
samples is:

    training_release  = 'dr12q';
    generate_dla_samples;

Processing spectra for DLA detection
------------------------------------

Finally, we may use our built model to compute the posterior
probability of containing a DLA along the line of sight as described
in the paper above.

The processing code requires the C helper function `voigt.c` to be
compiled to compute Voigt profiles quickly from MATLAB. This requires
the [`libcerf`](http://apps.jcns.fz-juelich.de/doku/sc/libcerf)
library to be installed. This is available
from [Homebrew-science](https://github.com/Homebrew/homebrew-science)
for OS X users. The code for this is:

    % in MATLAB
	mex voigt.c -lcerf

To perform a DLA search, we must specify a few things first. First, we
must specify which quasar emission model to use; to select the one
learned above, we may use

	% specify the learned quasar model to use
    training_release  = 'dr12q';
    training_set_name = 'dr9q_minus_concordance';

(the code will attempt to load the model from a file called
`data/[training_release]/processed/learned_qso_model_[training_set_name].mat`.)

Next, we must specify which spectra to use to compute the DLA model
prior Pr(M_DLA). Here we select all spectra that are:

* in DR9
* in the DR9 Lyman-alpha forest catalog, and
* not filtered by our filtering steps above.

These choices can be realized with:

    % specify the spectra to use for computing the DLA existence prior
    dla_catalog_name  = 'dr9q_concordance';
    prior_ind = ...
        [' prior_catalog.in_dr9 & ' ...
         ' prior_catalog.los_inds(dla_catalog_name) & ' ...
         '(prior_catalog.filter_flags == 0)'];

Next, we must specify which spectra to search for DLAs. Here we use
all DR12Q spectra that were not filtered:

    % specify the spectra to process
    release = 'dr12q';
    test_set_name = 'dr12q';
    test_ind = '(catalog.filter_flags == 0)';

Relevant parameters in `set_parameters` that can be tweaked if
desired, including function handles specifying the range of z_DLA to
search:

    % model prior parameters
	prior_z_qso_increase = kms_to_z(30000);       % use QSOs with z < (z_QSO + x) for prior

	% instrumental broadening parameters
	width = 3;                                    % width of Gaussian broadening (# pixels)
	pixel_spacing = 1e-4;                         % wavelength spacing of pixels in dex

	% DLA model parameters: absorber range and model
	num_lines = 3;                                % number of members of the Lyman series to use

	max_z_cut = kms_to_z(3000);                   % max z_DLA = z_QSO - max_z_cut
	max_z_dla = @(wavelengths, z_qso) ...         % determines maximum z_DLA to search
	    (max(wavelengths) / lya_wavelength - 1) - max_z_cut;

	min_z_cut = kms_to_z(3000);                   % min z_DLA = z_Ly∞ + min_z_cut
	min_z_dla = @(wavelengths, z_qso) ...         % determines minimum z_DLA to search
    max(min(wavelengths) / lya_wavelength - 1,                          ...
        observed_wavelengths(lyman_limit, z_qso) / lya_wavelength - 1 + ...
        min_z_cut);

When ready, the selected spectra can be processed with `process_qsos`.
This script will write the results in
`data/[release]/processed_qsos_[test_set_name].mat`.

The complete code for processing the spectra in MATLAB is:

    % produce catalog searching [Lyoo + 3000 km/s, Lya - 3000 km/s]
    set_parameters;

	% specify the learned quasar model to use
    training_release  = 'dr12q';
    training_set_name = 'dr9q_minus_concordance';

    % specify the spectra to use for computing the DLA existence prior
    dla_catalog_name  = 'dr9q_concordance';
	prior_ind = ...
        [' prior_catalog.in_dr9 & '             ...
         '(prior_catalog.filter_flags == 0) & ' ...
         ' prior_catalog.los_inds(dla_catalog_name)'];

    % specify the spectra to process
    release = 'dr12q';
    test_set_name = 'dr12q';
    test_ind = '(catalog.filter_flags == 0)';

	% process the spectra
    process_qsos;

Finally, we may create an ASCII catalog of the results if desired with
`generate_ascii_catalog`, e.g.:

    set_parameters;
    training_release  = 'dr12q';
    release = 'dr12q';
    test_set_name = 'dr12q';

    generate_ascii_catalog;
