#!/bin/bash

# download_spectra.sh: downloads DR12Q spectra from SDSS

base_directory='..'
pushd $base_directory/dr12q/spectra

wget -nc -nv -r -nH --cut-dirs=8 -i file_list -B http://data.sdss3.org/sas/dr12/boss/spectro/redux/

popd
