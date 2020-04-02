% set_lls_parameters: generates lls parameter and pdf

% set the prior parameters
% LLS model parameters: parameter samples
alpha               = 0.97;                    % weight of KDE component in mixture
min_lls_log_nhi     = 19.5;
uniform_min_log_nhi = 19.5;                   % range of column density samples    [cm⁻²]
uniform_max_log_nhi = 23.0;                   % from uniform distribution
fit_min_log_nhi     = 20.0;                   % range of column density samples    [cm⁻²]
fit_max_log_nhi     = 22.0;                   % from fit to log PDF
extrapolate_min_log_nhi = 19.5;               % normalization range for the extrapolated reagion

% load training catalog
catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

% generate quasirandom samples from p(normalized offset, log₁₀(N_HI))
rng('default');
sequence = scramble(haltonset(3), 'rr2');

% the first dimension can be used directly for the uniform prior over
% offsets
offset_samples  = sequence(1:num_dla_samples, 1)';

% we must transform the second dimension to have the correct marginal
% distribution for our chosen prior over column density, which is a
% mixture of a uniform distribution on log₁₀ N_HI and a distribution
% we fit to observed data

% uniform component of column density prior
u = makedist('uniform', ...
             'lower', uniform_min_log_nhi, ...
             'upper', uniform_max_log_nhi);

% extract observed log₁₀ N_HI samples from catalog
all_log_nhis = catalog.log_nhis(dla_catalog_name);
ind = cellfun(@(x) (~isempty(x)), all_log_nhis);
log_nhis = cat(1, all_log_nhis{ind});

% make a quadratic fit to the estimated log p(log₁₀ N_HI) over the
% specified range
x = linspace(fit_min_log_nhi, fit_max_log_nhi, 1e3);
kde_pdf = ksdensity(log_nhis, x);
f = polyfit(x, log(kde_pdf), 2);

% convert this to a PDF and normalize
% extrapolate the value at logNHI = 20 to logNHI < 20
unnormalized_pdf = ...
     @(nhi) ( exp(polyval(f,  nhi))       .*      heaviside( nhi - 20.03269 ) ... % 20.03269 is the analytical
          +   exp(polyval(f,  20.03269))  .* (1 - heaviside( nhi - 20.03269 )) ); % peak point of the quadratic
Z = integral(unnormalized_pdf, extrapolate_min_log_nhi, 25.0);                    % equation we had in polyfit

% create the PDF of the mixture between the unifrom distribution and
% the distribution fit to the data
normalized_pdf = @(nhi) ...
          alpha  * (unnormalized_pdf(nhi) / Z) + ...
     (1 - alpha) * (pdf(u, nhi));

% generate quasirandom samples from p(normalized offset, log₁₀(N_HI | M_lls))
lls_offset_samples = sequence(1:num_dla_samples, 3)'; % 1: zdla, 2: dla lognhi, 3: lls lognhi

% lls log_nhi samples from uniform(18, 20)
lls_log_nhi_samples = ...
      min_lls_log_nhi    + ...
    (fit_min_log_nhi - min_lls_log_nhi) * lls_offset_samples;
lls_nhi_samples = 10.^lls_log_nhi_samples;

% partition function for dla and lls based on integral ...
% Z_lls = integrate( p(logNHI), logNHI=[18, 20.3]) 
% Z_dla = integrate( p(logNHI), logNHI=[20.3, 25])
Z_lls = integral(normalized_pdf, min_lls_log_nhi, fit_min_log_nhi);
Z_dla = integral(normalized_pdf, fit_min_log_nhi, uniform_max_log_nhi);