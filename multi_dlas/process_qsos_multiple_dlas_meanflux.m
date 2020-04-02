% process_qsos_multiple_dlas_meanflux: run DLA detection algorithm on specified objects
% while using lower lognhi range (defined in set_lls_parameters.m) as an alternative model; 
% Note: model_posterior(quasar_ind, :) ... 
% = [p(no dla | D), p(lls | D), p(1 dla | D), p(2 dla | D), ...]
% also note that we should treat lls as no dla. 
%
% For higher order of DLA models, we consider the Occam's Razor
% effect due to normalisation in the higher dimensions of the parameter space.
%
% We implement exp(-optical_depth) to the mean-flux
%   µ(z) := µ * exp( - τ (1 + z)^β ) ; 
%   1 + z = lambda_obs / lambda_lya
% the prior of τ and β are taken from Kim, et al. (2007). 
%
% Nov 12, 2019: We added Lyman beta absorbers in the effective optical depth
%  optical_depth := τ (1 + z_a)^β + τ_b (1 + z_b)^β
%  τ_b     = τ f31 λ31 / ( f21 λ21 )
%  1 + z_a = λobs / λ_lya
%  1 + z_b = λobs / λ_lyb = λ_lya / λ_lyb  (1 + z_a) 
% 
% Nov 18, 2019: add all Lyman series to the effective optical depth
%   effective_optical_depth := ∑ τ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
%  where 
%   1 + z_i1 =  λobs / λ_i1 = λ_lya / λ_i1 *  (1 + z_a)
% Dec 25, 2019: add Lyman series to the noise variance training
%   s(z)     = 1 - exp(-effective_optical_depth) + c_0 
% 
% March 8, 2019: add additional Occam's razor factor between DLA models and null model:
%   P(DLAs | D) := P(DLAs | D) / num_dla_samples

% multi-dlas parameters
max_dlas = 4;
min_z_separation = kms_to_z(3000);

% the mean values of Kim's effective optical depth
prev_tau_0 = 0.0023;
prev_beta  = 3.65;

% load redshifts/DLA flags from training release
prior_catalog = ...
    load(sprintf('%s/catalog', processed_directory(training_release)));

if (ischar(prior_ind))
  prior_ind = eval(prior_ind);
end

prior.z_qsos  = prior_catalog.z_qsos(prior_ind);
prior.dla_ind = prior_catalog.dla_inds(dla_catalog_name);
prior.dla_ind = prior.dla_ind(prior_ind);

% filter out DLAs from prior catalog corresponding to region of spectrum below
% Ly∞ QSO rest
prior.z_dlas = prior_catalog.z_dlas(dla_catalog_name);
prior.z_dlas = prior.z_dlas(prior_ind);

for i = find(prior.dla_ind)'
  if (observed_wavelengths(lya_wavelength, prior.z_dlas{i}) < ...
      observed_wavelengths(lyman_limit,    prior.z_qsos(i)))
    prior.dla_ind(i) = false;
  end
end

prior = rmfield(prior, 'z_dlas');

% load QSO model from training release
variables_to_load = {'rest_wavelengths', 'mu', 'M', 'log_omega', ...
                     'log_c_0', 'log_tau_0', 'log_beta'};
load(sprintf('%s/learned_qso_model_lyseries_variance_kim_%s',             ...
             processed_directory(training_release), ...
             training_set_name),                    ...
     variables_to_load{:});

% load DLA samples from training release
variables_to_load = {'offset_samples', 'log_nhi_samples', 'nhi_samples'};
load(sprintf('%s/dla_samples_a03', processed_directory(training_release)), ...
     variables_to_load{:});

% load redshifts from catalog to process
catalog = load(sprintf('%s/catalog', processed_directory(release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
                     'all_pixel_mask'};
load(sprintf('%s/preloaded_qsos', processed_directory(release)), ...
     variables_to_load{:});

% enable processing specific QSOs via setting to_test_ind
if (ischar(test_ind))
  test_ind = eval(test_ind);
end

all_wavelengths    =    all_wavelengths(test_ind);
all_flux           =           all_flux(test_ind);
all_noise_variance = all_noise_variance(test_ind);
all_pixel_mask     =     all_pixel_mask(test_ind);

z_qsos = catalog.z_qsos(test_ind);

num_quasars = numel(z_qsos);

% preprocess model interpolants
mu_interpolator = ...
    griddedInterpolant(rest_wavelengths,        mu,        'linear');
M_interpolator = ...
    griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');
log_omega_interpolator = ...
    griddedInterpolant(rest_wavelengths,        log_omega, 'linear');

% initialize results
min_z_dlas                 = nan(num_quasars, 1);
max_z_dlas                 = nan(num_quasars, 1);
log_priors_no_dla          = nan(num_quasars, 1);
log_priors_dla             = nan(num_quasars, max_dlas);
log_likelihoods_no_dla     = nan(num_quasars, 1);
sample_log_likelihoods_dla = nan(num_quasars, num_dla_samples, max_dlas);
base_sample_inds           = zeros(num_quasars, num_dla_samples, max_dlas - 1, 'uint32');
log_likelihoods_dla        = nan(num_quasars, max_dlas);
log_posteriors_no_dla      = nan(num_quasars, 1);
log_posteriors_dla         = nan(num_quasars, max_dlas);

% initialize lls results
log_likelihoods_lls        = nan(num_quasars, 1);
log_posteriors_lls         = nan(num_quasars, 1);
log_priors_lls             = nan(num_quasars, 1);
sample_log_likelihoods_lls = nan(num_quasars, num_dla_samples);

% save maps: add the initilizations of MAP values
% N * (1~k models) * (1~k MAP dlas)
MAP_z_dlas   = nan(num_quasars, max_dlas, max_dlas);
MAP_log_nhis = nan(num_quasars, max_dlas, max_dlas);
MAP_inds     = nan(num_quasars, max_dlas, max_dlas);


c_0   = exp(log_c_0);
tau_0 = exp(log_tau_0);
beta  = exp(log_beta);

% handle the inds of empty spectra
all_exceptions = nan(num_quasars, 1);

for quasar_ind = 1:num_quasars
  tic;
  rng('default');  % random number should be set for each qso run

  % initialize an empty array for this sample log likelihood
  this_sample_log_likelihoods_dla = nan(num_dla_samples, max_dlas);

  z_qso = z_qsos(quasar_ind);

  fprintf('processing quasar %i/%i (z_QSO = %0.4f) ...', ...
        quasar_ind, num_quasars, z_qso);

  this_wavelengths    =    all_wavelengths{quasar_ind};
  this_flux           =           all_flux{quasar_ind};
  this_noise_variance = all_noise_variance{quasar_ind};
  this_pixel_mask     =     all_pixel_mask{quasar_ind};

  % convert to QSO rest frame
  this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso);

  ind = (this_rest_wavelengths >= min_lambda) & ...
        (this_rest_wavelengths <= max_lambda);

  % keep complete copy of equally spaced wavelengths for absorption
  % computation
  this_unmasked_wavelengths = this_wavelengths(ind);

  ind = ind & (~this_pixel_mask);

  this_wavelengths      =      this_wavelengths(ind);
  this_rest_wavelengths = this_rest_wavelengths(ind);
  this_flux             =             this_flux(ind);
  this_noise_variance   =   this_noise_variance(ind);

  this_lya_zs = ...
      (this_wavelengths - lya_wavelength) / ...
      lya_wavelength;

  % To count the effect of Lyman series from higher z,
  % we compute the absorbers' redshifts for all members of the series
  this_lyseries_zs = nan(numel(this_wavelengths), num_forest_lines);
  
  for l = 1:num_forest_lines
    this_lyseries_zs(:, l) = ...
      (this_wavelengths - all_transition_wavelengths(l)) / ...
      all_transition_wavelengths(l);    
  end
      
  % DLA existence prior
  less_ind = (prior.z_qsos < (z_qso + prior_z_qso_increase));

  this_num_dlas    = nnz(prior.dla_ind(less_ind));
  this_num_quasars = nnz(less_ind);
  this_p_dlas      = (this_num_dlas / this_num_quasars).^(1:max_dlas);

  for i = 1:max_dlas
    this_p_dlas(i) = this_p_dlas(i) - sum(this_p_dlas((i + 1):end));
    log_priors_dla(quasar_ind, i) = log(this_p_dlas(i));
  end

  % lls priors : assume extrapolated log_nhi prior for lls
  % lls prior = M / N * Z_lls / Z_dla
  log_priors_lls(quasar_ind) = ...
    log(this_num_dlas) - log(this_num_quasars) + ...
    log(Z_lls)         - log(Z_dla);

  % no dla priors : subtract from dla and lls priors
  % no dla prior = 1 - M / N - M / N * Z_lls / Z_dla
  log_priors_no_dla(quasar_ind) = ...
      log(this_num_quasars - this_num_dlas - Z_lls * this_num_dlas / Z_dla) ...
      - log(this_num_quasars);

  fprintf_debug('\n');
  for i = 1:max_dlas
    fprintf_debug(' ...     p(%i  DLAs | z_QSO)       : %0.3f\n', i, this_p_dlas(i));
  end
  fprintf_debug(' ...     p(no DLA  | z_QSO)       : %0.3f\n', exp(log_priors_no_dla(quasar_ind)) );
  fprintf_debug(' ...     p(sub DLA | z_QSO)       : %0.3f\n', exp(log_priors_lls(quasar_ind)) );

  % interpolate model onto given wavelengths
  % error will appear if the input spectrum is empty
  try
    this_mu = mu_interpolator( this_rest_wavelengths);
    this_M  =  M_interpolator({this_rest_wavelengths, 1:k});
  catch me
    if (strcmp(me.identifier, 'MATLAB:griddedInterpolant:NonVecCompVecErrId'))
      all_exceptions(quasar_ind, 1) = 1;
      fprintf(' took %0.3fs.\n', toc);      
      continue
    else
      rethrow(me)
    end
  end

  this_log_omega = log_omega_interpolator(this_rest_wavelengths);
  this_omega2 = exp(2 * this_log_omega);

  % Lyman series absorption effect for the noise variance
  % note: this noise variance must be trained on the same number of members of Lyman series
  lya_optical_depth = tau_0 .* (1 + this_lya_zs).^beta;

  for l = 2:num_forest_lines
    lyman_1pz = all_transition_wavelengths(1) .* (1 + this_lya_zs) ...
        ./ all_transition_wavelengths(l);

    % only include the Lyman series with absorber redshifts lower than z_qso
    indicator = lyman_1pz <= (1 + z_qso);
    lyman_1pz = lyman_1pz .* indicator;

    tau = tau_0 * all_transition_wavelengths(l) * all_oscillator_strengths(l) ...
        / (  all_transition_wavelengths(1) * all_oscillator_strengths(1) );

    lya_optical_depth = lya_optical_depth + tau .* lyman_1pz.^beta;
  end

  this_scaling_factor = 1 - exp( -lya_optical_depth ) + c_0;

  this_omega2 = this_omega2 .* this_scaling_factor.^2;

  % Lyman series absorption effect on the mean-flux
  % apply the lya_absorption after the interpolation because NaN will appear in this_mu
  total_optical_depth = nan(numel(this_wavelengths), num_forest_lines);

  for l = 1:num_forest_lines
    % calculate the oscillator strength for this lyman series member
    this_tau_0 = prev_tau_0 * ...
      all_oscillator_strengths(l)   / lya_oscillator_strength * ...
      all_transition_wavelengths(l) / lya_wavelength;
    
    total_optical_depth(:, l) = ...
      this_tau_0 .* ( (1 + this_lyseries_zs(:, l)).^prev_beta );

    % indicator function: z absorbers <= z_qso
    if l > 1
      indicator = this_lyseries_zs(:, l) > z_qso;
      total_optical_depth(indicator, l) = nan;
    end
  end

  lya_absorption = exp(- nansum(total_optical_depth, 2) );
  
  this_mu = this_mu .* lya_absorption;
  this_M  = this_M  .* lya_absorption;

  % re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
  % now the null model likelihood is:
  % p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
  this_omega2 = this_omega2 .* lya_absorption.^2;

  % baseline: probability of no DLA model
  log_likelihoods_no_dla(quasar_ind) = ...
      log_mvnpdf_low_rank(this_flux, this_mu, this_M, ...
          this_omega2 + this_noise_variance);

  log_posteriors_no_dla(quasar_ind) = ...
      log_priors_no_dla(quasar_ind) + log_likelihoods_no_dla(quasar_ind);

  fprintf_debug(' ... log p(  D  | z_QSO, no DLA ) : %0.2f\n', ...
                log_likelihoods_no_dla(quasar_ind));

  min_z_dlas(quasar_ind) = min_z_dla(this_wavelengths, z_qso);
  max_z_dlas(quasar_ind) = max_z_dla(this_wavelengths, z_qso);

  sample_z_dlas = ...
       min_z_dlas(quasar_ind) +  ...
      (max_z_dlas(quasar_ind) - min_z_dlas(quasar_ind)) * offset_samples;

  this_base_sample_inds = zeros(max_dlas - 1, num_dla_samples, 'uint32');

  % save maps: make extract MAP values much easier
  % the only difference is including the 1st model inds
  this_MAP_base_sample_inds       = zeros(max_dlas, num_dla_samples, 'uint32');
  this_MAP_base_sample_inds(1, :) = 1:num_dla_samples;

  % ensure enough pixels are on either side for convolving with
  % instrument profile
  padded_wavelengths = ...
      [logspace(log10(min(this_unmasked_wavelengths)) - width * pixel_spacing, ...
                log10(min(this_unmasked_wavelengths)) - pixel_spacing,         ...
                width)';                                                       ...
       this_unmasked_wavelengths;                                              ...
       logspace(log10(max(this_unmasked_wavelengths)) + pixel_spacing,         ...
                log10(max(this_unmasked_wavelengths)) + width * pixel_spacing, ...
                width)'                                                        ...
      ];

  % to retain only unmasked pixels from computed absorption profile
  mask_ind = (~this_pixel_mask(ind));

  for num_dlas = 1:max_dlas
    % compute probabilities under DLA model for each of the sampled
    % (normalized offset, log(N HI)) pairs
    parfor i = 1:num_dla_samples
      % absorption corresponding to this sample
      absorption = voigt(padded_wavelengths, sample_z_dlas(i), ...
                         nhi_samples(i), num_lines);

      % absorption corresponding to other DLAs in multiple DLA samples
      for j = 1:(num_dlas - 1)
        k = this_base_sample_inds(j, i);
        absorption = absorption .* ...
            voigt(padded_wavelengths, sample_z_dlas(k), ...
                  nhi_samples(k), num_lines);
      end

      absorption = absorption(mask_ind);

      dla_mu     = this_mu     .* absorption;
      dla_M      = this_M      .* absorption;
      dla_omega2 = this_omega2 .* absorption.^2;

      this_sample_log_likelihoods_dla(i, num_dlas) = ...
        log_mvnpdf_low_rank(this_flux, dla_mu, dla_M, ...
            dla_omega2 + this_noise_variance) - log(num_dla_samples);
            % additional occams razor

      % compute lls model
      if (num_dlas == 1)
        % absorption with lls column density
        absorption = voigt(padded_wavelengths, sample_z_dlas(i), ...
          lls_nhi_samples(i), num_lines);

        absorption = absorption(mask_ind);

        lls_mu     = this_mu     .* absorption;
        lls_M      = this_M      .* absorption;
        lls_omega2 = this_omega2 .* absorption.^2;

        sample_log_likelihoods_lls(quasar_ind, i) = ...
          log_mvnpdf_low_rank(this_flux, lls_mu, lls_M, ...
              lls_omega2 + this_noise_variance) - log(num_dla_samples);
              % additional occams razor
      end
    end

    % check if any pair of dlas in this sample is too close this has to
    % happen outside the parfor because "continue" slows things down
    % dramatically
    if (num_dlas > 1)
      ind = this_base_sample_inds(1:(num_dlas - 1), :);
      all_z_dlas   = [sample_z_dlas; sample_z_dlas(ind)];
      all_log_nhis = [log_nhi_samples; log_nhi_samples(ind)]; 

      ind = any(diff(sort(all_z_dlas)) < min_z_separation, 1);
      this_sample_log_likelihoods_dla(ind, num_dlas) = nan;

    elseif (num_dlas == 1)
      all_z_dlas   = sample_z_dlas;
      all_log_nhis = log_nhi_samples;

    end

    max_log_likelihood = ...
        nanmax(this_sample_log_likelihoods_dla(:, num_dlas));

    sample_probabilities = ...
        exp(this_sample_log_likelihoods_dla(:, num_dlas) - ...
            max_log_likelihood);

    log_likelihoods_dla(quasar_ind, num_dlas) = ...
        max_log_likelihood + log(nanmean(sample_probabilities)) ...
        - log( num_dla_samples ) * (num_dlas - 1); % occam's razor

    log_posteriors_dla(quasar_ind, num_dlas) = ...
        log_priors_dla(quasar_ind, num_dlas) + ...
        log_likelihoods_dla(quasar_ind, num_dlas);

    % compute lls log posterior          
    if (num_dlas == 1)
      max_log_likelihood_lls = ...
        nanmax(sample_log_likelihoods_lls(quasar_ind, :));
      
      sample_probabilities_lls = ...
        exp(sample_log_likelihoods_lls(quasar_ind, :) - ...
            max_log_likelihood_lls);
      
      log_likelihoods_lls(quasar_ind) = ...
        max_log_likelihood_lls + log(nanmean(sample_probabilities_lls)) ...
        - log( num_dla_samples ) * (num_dlas - 1); % occam's razor

      log_posteriors_lls(quasar_ind) = ...
          log_priors_lls(quasar_ind) + ...
          log_likelihoods_lls(quasar_ind);

      fprintf_debug(' ... log p(D | z_QSO, sub DLA) : %0.2f\n', ...
          log_likelihoods_lls(quasar_ind) );
      fprintf_debug(' ... log p(sub DLA | D, z_QSO) : %0.2f\n', ...
          log_posteriors_lls(quasar_ind) );    
    end
        
    % save map: extract MAP values of z_dla and log_nhi
    [~, maxidx] = nanmax(this_sample_log_likelihoods_dla(:, num_dlas), [], 1);

    MAP_inds(quasar_ind, num_dlas, 1:num_dlas)     = [maxidx; this_base_sample_inds(1:(num_dlas - 1), maxidx)];

    % save map: save the MAP values to the array
    MAP_z_dlas(quasar_ind, num_dlas, 1:num_dlas)   = all_z_dlas(:, maxidx);
    MAP_log_nhis(quasar_ind, num_dlas, 1:num_dlas) = all_log_nhis(:, maxidx);
    
    fprintf_debug(' ... log p(D | z_QSO, %i DLAs) : %0.2f\n', ...
                  num_dlas, log_likelihoods_dla(quasar_ind, num_dlas));
    fprintf_debug(' ... log p(%i DLAs | D, z_QSO) : %0.2f\n', ...
                  num_dlas, log_posteriors_dla(quasar_ind, num_dlas));

    if (num_dlas == max_dlas)
      break;
    end

    % if p(D | z_QSO, num_dlas DLA) is NaN, then
    % finish the loop. 
    % It's usually because p(D | z_QSO, no DLA) is very high, so
    % the higher order DLA model likelihoods already uderflowed
    if isnan(log_likelihoods_dla(quasar_ind, num_dlas))
      fprintf('Finish the loop earlier because NaN value in log p(D | z_QSO, %i DLAs) : %0.2f\n', ...
        num_dlas, log_likelihoods_dla(quasar_ind, num_dlas));
      break;
    end

    % avoid nan values in the randsample weights
    nanind = isnan(sample_probabilities);
    W = sample_probabilities;
    W(nanind) = double(0);

    this_base_sample_inds(num_dlas, :) = ...
        uint32(randsample(num_dla_samples, num_dla_samples, true, W)');
  end

  % exclude to save memory
  base_sample_inds(quasar_ind, :, :)           = this_base_sample_inds';
  sample_log_likelihoods_dla(quasar_ind, :, :) = this_sample_log_likelihoods_dla(:, :);

  fprintf(' took %0.3fs.\n', toc);
end

max_log_posteriors = ...
    max([log_posteriors_no_dla, log_posteriors_lls, log_posteriors_dla], [], 2);

model_posteriors = ...
    exp(bsxfun(@minus, ...
               [log_posteriors_no_dla, log_posteriors_lls, log_posteriors_dla], ...
               max_log_posteriors));

model_posteriors = ...
    bsxfun(@times, model_posteriors, 1 ./ sum(model_posteriors, 2));

p_no_dlas = model_posteriors(:, 1);
p_lls     = model_posteriors(:, 2);
p_dlas    = 1 - p_no_dlas - p_lls;

% save results
variables_to_save = {'training_release', 'training_set_name', ...
                     'dla_catalog_name', 'prior_ind', 'release', ...
                     'test_set_name', 'prior_z_qso_increase', 'k', ...
                     'normalization_min_lambda', 'normalization_max_lambda', ...
                     'min_z_cut', 'max_z_cut', 'num_dla_samples', ...
                     'num_lines', 'min_z_dlas', 'max_z_dlas', ...
                     'sample_log_likelihoods_dla', 'base_sample_inds', ...
                     'log_priors_no_dla', 'log_priors_dla', 'log_priors_lls', ...
                     'log_likelihoods_no_dla', 'MAP_z_dlas', 'MAP_log_nhis', ...
                     'log_likelihoods_dla', 'log_likelihoods_lls', ...
                     'log_posteriors_no_dla', 'log_posteriors_dla', 'log_posteriors_lls', ...
                     'model_posteriors', 'p_no_dlas', 'p_dlas', 'p_lls', ...
                     'all_exceptions', 'sample_log_likelihoods_lls'};

if (exist('test_ind', 'var'))
  filename = sprintf('%s/processed_qsos_multi_meanflux%s', ...
                     processed_directory(release), ...
                     test_set_name);

  variables_to_save{end + 1} = 'test_ind';
else
  filename = sprintf('%s/processed_qsos_multi_meanflux', ...
                     processed_directory(release));
end

save(filename, variables_to_save{:}, '-v7.3');
