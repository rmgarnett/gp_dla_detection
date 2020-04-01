% zwarning_patch.m : add zwarning flag back to the existing catalog.mat

% load QSO catalogs
release = 'dr12q';
dr12_catalog = ...
    fitsread(sprintf('%s/DR12Q.fits', distfiles_directory(release)), ...
             'binarytable');

% load QSO catalog
variables_to_load = {'filter_flags'};
load(sprintf('%s/catalog', processed_directory(release)), ...
    variables_to_load{:});

% extract ZWARNING flag from the catalog
zwarning =  dr12_catalog{11};

% filtering bit 4: ZWARNING
ind = (zwarning > 0);
%% but include `MANY_OUTLIERS` in our samples (bit: 1000)
ind_many_outliers      = (zwarning == bin2dec('10000'));
ind(ind_many_outliers) = 0;
filter_flags(ind) = bitset(filter_flags(ind), 5, true);

% write new filter flags to catalog
variables_to_save = {'filter_flags', 'zwarning'};
save(sprintf('%s/catalog', processed_directory(release)), ...
     variables_to_save{:}, '-append');
