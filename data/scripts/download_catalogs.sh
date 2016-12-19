#!/bin/bash

# downlaod_catalogs.sh: downloads SDSS DR9Q, DR10Q, and DR12Q
# catalogs, as well as corresponding previously compiled DLA catalogs

base_directory='..'
pushd $base_directory

# DR9Q
directory='dr9q'

mkdir -p $directory/distfiles
pushd $directory/distfiles
filename='DR9Q.fits'
wget http://data.sdss3.org/sas/dr9/env/BOSS_QSO/DR9Q/$filename -O $filename
popd

# DR10Q
directory='dr10q'

mkdir -p $directory/distfiles
pushd $directory/distfiles
filename='DR10Q_v2.fits'
wget http://data.sdss3.org/sas/dr10/boss/qso/DR10Q/$filename -O $filename
popd

# DR12Q
directory='dr12q'

mkdir -p $directory/spectra $directory/processed $directory/distfiles
pushd $directory/distfiles
filename='DR12Q.fits'
wget http://data.sdss3.org/sas/dr12/boss/qso/DR12Q/$filename -O $filename
popd

# DLA catalogs
directory='dla_catalogs'
mkdir -p $directory
pushd $directory

  # concordance catalog from BOSS DR9 Lyman-alpha forest catalog
  catalog='dr9q_concordance'
  mkdir -p $catalog/distfiles $catalog/processed
  pushd $catalog/distfiles
    filename='BOSSLyaDR9_cat.txt'
    wget http://data.sdss3.org/sas/dr9/boss/lya/cat/$filename -O $filename
  popd
  pushd $catalog
    gawk '(NR > 1 && $15 > 0) {print $4, $15, $16}' distfiles/$filename > processed/dla_catalog
    gawk '(NR > 1)            {print $4}'           distfiles/$filename > processed/los_catalog
  popd

  # DR12Q DLA catalog from Noterdaeme, et al.
  catalog='dr12q_noterdaeme'
  mkdir -p $catalog/distfiles $catalog/processed
  pushd $catalog/distfiles
    filename='DLA_DR12_v2.tgz'
    wget http://www2.iap.fr/users/noterdae/DLA/$filename -O $filename
    tar xvf $filename
  popd
  pushd $catalog
    gawk '(NR > 2 && NF > 0) {print $1, $10, $11}' distfiles/DLA_DR12_v2.dat > processed/dla_catalog
    gawk '(NR > 2 && NF > 0) {print $1}'           distfiles/LOS_DR12_v2.dat > processed/los_catalog
  popd

  # DR12Q DLA visual survey, extracted from Noterdaeme, et al.
  catalog='dr12q_visual'
  mkdir -p $catalog/distfiles $catalog/processed
  pushd $catalog/distfiles
    filename='DLA_DR12_v2.tgz'
    wget http://www2.iap.fr/users/noterdae/DLA/$filename -O $filename
    tar xvf $filename
  popd
  # redshifts and column densities are not available for visual
  # survey, so fill in with z_QSO and DLA threshold density
  # (log_10 N_HI = 20.3)
  pushd $catalog
    gawk '(NR > 2 && NF > 0 && $6) {print $1, $5, 20.3}' distfiles/LOS_DR12_v2.dat > processed/dla_catalog
    gawk '(NR > 2 && NF > 0) {print $1}'                 distfiles/LOS_DR12_v2.dat > processed/los_catalog
  popd

popd
