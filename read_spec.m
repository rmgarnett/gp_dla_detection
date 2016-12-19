% read_spec: loads data from SDSS DR12Q coadded "speclite" FITS file;
% see
% https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
% for a complete description of the data format

function [wavelengths, flux, noise_variance, pixel_mask] = read_spec(filename)

  % mask bits to consider
  BRIGHTSKY = 24;

  measurements = fitsread(filename, ...
          'binarytable',  1, ...
          'tablecolumns', 1:4);

  % coadded calibrated flux  10^-17 erg s^-1 cm^-2 A^-1
  flux = measurements{1};

  % log_10 wavelength        log A
  log_wavelengths = measurements{2};

  % inverse noise variance of flux measurements
  inverse_noise_variance = measurements{3};

  % "and" mask
  and_mask = measurements{4};

  % convert log_10 wavelengths to wavelengths
  wavelengths = 10.^log_wavelengths;

  % derive noise variance
  noise_variance = 1 ./ (inverse_noise_variance);

  % derive bad pixel mask, remove pixels considered very bad
  % (FULLREJECT, NOSKY, NODATA); additionally remove pixels with
  % BRIGHTSKY set
  pixel_mask = ...
      (inverse_noise_variance == 0) | ...
      (bitget(and_mask, BRIGHTSKY));

end