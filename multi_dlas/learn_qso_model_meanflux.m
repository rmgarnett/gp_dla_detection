% learn_qso_model_meanflux: fits GP to training catalog via maximum likelihood
%   mean vector is modelled to change with the mean-flux with the implemetation of Lyman-series forest 
%   using Kim's mean values. Also, noise variance is modelled to include Lyman series.
%
%   lift     mean-flux mu in the learning script via median(   y .* exp( + optical_depth ) ),
%   suppress mean-flux mu in the processing script via mu' := mu .* exp( - optical_depth );

rng('default');

% load catalog
catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
                     'all_pixel_mask'};
load(sprintf('%s/preloaded_qsos', processed_directory(training_release)), ...
     variables_to_load{:});

% determine which spectra to use for training; allow string value for
% train_ind
if (ischar(train_ind))
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

lya_1pzs             = nan(num_quasars, num_rest_pixels);
all_lyman_1pzs       = nan(num_forest_lines, num_quasars, num_rest_pixels);
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

  lya_1pzs(i, :) = ...
      interp1(this_rest_wavelengths, ...
              1 + (this_wavelengths - lya_wavelength) / lya_wavelength, ...
              rest_wavelengths);

  % incldue all members in Lyman series to the forest
  for j = 1:num_forest_lines
    this_transition_wavelength = all_transition_wavelengths(j);

    all_lyman_1pzs(j, i, :) = ...
      interp1(this_rest_wavelengths, ...
              1 + (this_wavelengths - this_transition_wavelength) / this_transition_wavelength, ... 
              rest_wavelengths);

    if j > 1
      % indicator function: z absorbers <= z_qso
      indicator = all_lyman_1pzs(j, i, :) <= (1 + z_qso);

      all_lyman_1pzs(j, i, :) = all_lyman_1pzs(j, i, :) .* indicator;    
    end
  end

  rest_fluxes(i, :) = ...
      interp1(this_rest_wavelengths, this_flux,           rest_wavelengths);

  rest_noise_variances(i, :) = ...
      interp1(this_rest_wavelengths, this_noise_variance, rest_wavelengths);
end
clear('all_wavelengths', 'all_flux', 'all_noise_variance', 'all_pixel_mask');

% mask noisy pixels
ind = (rest_noise_variances > max_noise_variance);
lya_1pzs(ind)             = nan;
rest_fluxes(ind)          = nan;
rest_noise_variances(ind) = nan;
for i = 1:num_quasars
  for j = 1:num_forest_lines
    all_lyman_1pzs(j, i, ind(i, :))  = nan;
  end
end

% find empirical mean vector:
% reverse the rest_fluxes back to the fluxes before encountering Lyα forest
prev_tau_0 = 0.0023; % Kim et al. (2007) priors
prev_beta  = 3.65;

rest_fluxes_div_exp1pz      = nan(num_quasars, num_rest_pixels);
rest_noise_variances_exp1pz = nan(num_quasars, num_rest_pixels);

for i = 1:num_quasars
  % compute the total optical depth from all Lyman series members
  total_optical_depth = nan(num_forest_lines, num_rest_pixels);

  for j = 1:num_forest_lines
    % calculate the oscillator strengths for Lyman series
    this_tau_0 = prev_tau_0 * ...
      all_oscillator_strengths(j)   / lya_oscillator_strength * ...
      all_transition_wavelengths(j) / lya_wavelength;
    
    % remove the leading dimension
    this_lyman_1pzs = squeeze(all_lyman_1pzs(j, i, :))'; % (1, num_rest_pixels)

    total_optical_depth(j, :) = this_tau_0 .* (this_lyman_1pzs.^prev_beta);
  end

  lya_absorption = exp(- nansum(total_optical_depth, 1) );

  % We have to reverse the effect of Lyα for both mean-flux and observational noise
  rest_fluxes_div_exp1pz(i, :)      = rest_fluxes(i, :) ./ lya_absorption;
  rest_noise_variances_exp1pz(i, :) = rest_noise_variances(i, :) ./ (lya_absorption.^2);
end

clear('all_lyman_1pzs');

mu = nanmean(rest_fluxes_div_exp1pz);
centered_rest_fluxes = bsxfun(@minus, rest_fluxes_div_exp1pz, mu);
clear('rest_fluxes', 'rest_fluxes_div_exp1pz');

% get top-k PCA vectors to initialize M
[coefficients, ~, latent] = ...
    pca(centered_rest_fluxes, ...
        'numcomponents', k, ...
        'rows',          'complete');

objective_function = @(x) objective_lyseries(x, centered_rest_fluxes, lya_1pzs, ...
        rest_noise_variances_exp1pz, num_forest_lines, all_transition_wavelengths, ...
        all_oscillator_strengths);

% initialize A to top-k PCA components of non-DLA-containing spectra
initial_M = bsxfun(@times, coefficients(:, 1:k), sqrt(latent(1:k))');

% initialize log omega to log of elementwise sample standard deviation
initial_log_omega = log(nanstd(centered_rest_fluxes));

initial_log_c_0   = log(initial_c_0);
initial_log_tau_0 = log(initial_tau_0);
initial_log_beta  = log(initial_beta);

initial_x = [initial_M(:);         ...
             initial_log_omega(:); ...
             initial_log_c_0;      ...
             initial_log_tau_0;    ...
             initial_log_beta];

% maximize likelihood via L-BFGS
[x, log_likelihood, ~, minFunc_output] = ...
    minFunc(objective_function, initial_x, minFunc_options);

ind = (1:(num_rest_pixels * k));
M = reshape(x(ind), [num_rest_pixels, k]);

ind = ((num_rest_pixels * k + 1):(num_rest_pixels * (k + 1)));
log_omega = x(ind)';

log_c_0   = x(end - 2);
log_tau_0 = x(end - 1);
log_beta  = x(end);

variables_to_save = {'training_release', 'train_ind', 'max_noise_variance', ...
                     'minFunc_options', 'rest_wavelengths', 'mu', ...
                     'initial_M', 'initial_log_omega', 'initial_log_c_0', ...
                     'initial_tau_0', 'initial_beta',  'M', 'log_omega', ...
                     'log_c_0', 'log_tau_0', 'log_beta', 'log_likelihood', ...
                     'minFunc_output'};

save(sprintf('%s/learned_qso_model_lyseries_variance_kim_%s',             ...
             processed_directory(training_release), ...
             training_set_name), ...
     variables_to_save{:}, '-v7.3');
