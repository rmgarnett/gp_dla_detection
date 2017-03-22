% set_parameters: sets various parameters for the DLA detection
% pipeline

% physical constants
lya_wavelength = 1215.6701;                   % Lyman alpha transition wavelength  A
lyb_wavelength = 1025.7223;                   % Lyman beta  transition wavelength  A
lyman_limit    =  911.7633;                   % Lyman limit wavelength             A
speed_of_light = 299792458;                   % speed of light                     m s^-1

% converts relative velocity in km s^-1 to redshift difference
kms_to_z = @(kms) (kms * 1000) / speed_of_light;

% utility functions for redshifting
emitted_wavelengths = ...
    @(observed_wavelengths, z) (observed_wavelengths / (1 + z));

observed_wavelengths = ...
    @(emitted_wavelengths,  z) ( emitted_wavelengths * (1 + z));

% preprocessing parameters
z_qso_cut      = 2.15;                        % filter out QSOs with z less than this threshold
min_num_pixels = 200;                         % minimum number of non-masked pixels

% normalization parameters
normalization_min_lambda = 1310;              % range of rest wavelengths to use   A
normalization_max_lambda = 1325;              %   for flux normalization           A

% file loading parameters
loading_min_lambda = 910;                     % range of rest wavelengths to load  A
loading_max_lambda = 1217;                    %                                    A

% null model parameters
min_lambda         =  911.75;                 % range of rest wavelengths to       A
max_lambda         = 1215.75;                 %   model
dlambda            =    0.25;                 % separation of wavelength grid      A
k                  = 20;                      % rank of non-diagonal contribution
max_noise_variance = 1^2;                     % maximum pixel noise allowed during model training

% BFGS parameters
minFunc_options =               ...           % optimization options for model fitting
    struct('MaxIter',     2000, ...
           'MaxFunEvals', 4000);

% DLA model parameters: parameter samples
num_dla_samples     = 10000;                  % number of parameter samples
alpha               = 0.9;                    % weight of KDE component in mixture
uniform_min_log_nhi = 20.0;                   % range of column density samples    [cm^-2]
uniform_max_log_nhi = 23.0;                   % from uniform distribution
fit_min_log_nhi     = 20.0;                   % range of column density samples    [cm^-2]
fit_max_log_nhi     = 22.0;                   % from fit to log PDF

% model prior parameters
prior_z_qso_increase = kms_to_z(30000);       % use QSOs with z < (z_QSO + x) for prior

% DLA model parameters: absorber range and model
num_lines = 3;                                % number of members of the Lyman series to use
max_z_cut = kms_to_z(3000);                   % max z_DLA = z_QSO - max_z_cut
max_z_dla = @(wavelengths, z_qso) ...         % determines maximum z_DLA to search
    (max(wavelengths) / lya_wavelength - 1) - max_z_cut;
min_z_cut = kms_to_z(3000);                   % min z_DLA = z_Lyoo + min_z_cut
min_z_dla = @(wavelengths, z_qso) ...         % determines minimum z_DLA to search
    max(min(wavelengths) / lya_wavelength - 1,                          ...
        observed_wavelengths(lyman_limit, z_qso) / lya_wavelength - 1 + ...
        min_z_cut);

% base directory for all data
base_directory = '~/work/gdd/data';

% utility functions for identifying various directories
distfiles_directory = @(release) ...
    sprintf('%s/%s/distfiles', base_directory, release);

spectra_directory   = @(release) ...
    sprintf('%s/%s/spectra',   base_directory, release);

processed_directory = @(release) ...
    sprintf('%s/%s/processed', base_directory, release);

dla_catalog_directory = @(name) ...
    sprintf('%s/dla_catalogs/%s/processed', base_directory, name);

% replace with @(varargin) (fprintf(varargin{:})) to show debug statements
fprintf_debug = @(varargin) ([]);
