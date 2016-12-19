% learn_qso_model: fits GP to training catalog via maximum likelihood

rng('default');

% load catalog
catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
                     'all_pixel_mask'};
load(sprintf('%s/preloaded_qsos', processed_directory(training_release)), ...
     variables_to_load{:});

% determine which spectra to use for training; allow string value for
% train_ind, guaranteeing that the variable "num_quasars" exists
if (ischar(train_ind))
  num_quasars = numel(catalog.z_qsos);
  train_ind = eval(train_ind);
end

% select training vectors
all_wavelengths    =    all_wavelengths(train_ind, :);
all_flux           =           all_flux(train_ind, :);
all_noise_variance = all_noise_variance(train_ind, :);
all_pixel_mask     =     all_pixel_mask(train_ind, :);
z_qsos             =     catalog.z_qsos(train_ind);

num_quasars = numel(z_qsos);

rest_wavelengths = (min_lambda:dlambda:max_lambda);

num_rest_pixels = numel(rest_wavelengths);
rest_fluxes          = nan(num_quasars, num_rest_pixels);
rest_noise_variances = nan(num_quasars, num_rest_pixels);

% interpolate quasars onto chosen rest wavelength grid
for i = 1:num_quasars
  z_qso = z_qsos(i);

  this_wavelengths    =    all_wavelengths{i}';
  this_flux           =           all_flux{i}';
  this_noise_variance = all_noise_variance{i}';
  this_pixel_mask     =     all_pixel_mask{i}';

  this_flux(this_pixel_mask)           = nan;
  this_noise_variance(this_pixel_mask) = nan;

  this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso);

  rest_fluxes(i, :) = ...
      interp1(this_rest_wavelengths, this_flux,           rest_wavelengths);

  rest_noise_variances(i, :) = ...
      interp1(this_rest_wavelengths, this_noise_variance, rest_wavelengths);
end
clear('all_wavelengths', 'all_flux', 'all_noise_variance', 'all_pixel_mask');

% mask noisy pixels
ind = (rest_noise_variances > max_noise_variance);
rest_fluxes(ind)          = nan;
rest_noise_variances(ind) = nan;

% find empirical mean vector and center data
mu = nanmean(rest_fluxes);
centered_rest_fluxes = bsxfun(@minus, rest_fluxes, mu);
clear('rest_fluxes');

% get top-k PCA vectors to initialize M
[coefficients, ~, latent] = ...
    pca(centered_rest_fluxes, ...
        'numcomponents', k, ...
        'rows',          'pairwise');

objective_function = @(x) objective(x, centered_rest_fluxes, rest_noise_variances);

% initialize A to top-k PCA components of non-DLA-containing spectra
initial_M = bsxfun(@times, coefficients(:, 1:k), sqrt(latent(1:k))');

% initialize log sigma to log of elementwise sample standard deviation
initial_log_sigma = log(nanstd(centered_rest_fluxes));

initial_x = [initial_M(:); initial_log_sigma(:)];

% maximize likelihood via L-BFGS
[x, log_likelihood, ~, minFunc_output] = ...
    minFunc(objective_function, initial_x, minFunc_options);

ind = (1:(num_rest_pixels * k));
M = reshape(x(ind), [num_rest_pixels, k]);

ind = ((num_rest_pixels * k + 1):(num_rest_pixels * (k + 1)));
log_sigma = x(ind)';

variables_to_save = {'training_release', 'train_ind', 'rest_wavelengths', ...
                     'max_noise_variance', 'minFunc_options', 'initial_M', ...
                     'initial_log_sigma', 'mu', 'M', 'log_sigma', ...
                     'log_likelihood', 'minFunc_output'};
save(sprintf('%s/learned_qso_model_%s',             ...
             processed_directory(training_release), ...
             training_set_name), ...
     variables_to_save{:}, '-v7.3');
