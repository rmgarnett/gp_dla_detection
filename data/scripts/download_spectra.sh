#!/bin/bash

# download_spectra.sh: downloads DR12Q spectra from SDSS

base_directory='..'
pushd $base_directory/dr12q/spectra

rsync --info=progress2 -h --no-motd --files-from=file_list rsync://data.sdss.org/dr12/boss/spectro/redux/ . 2> /dev/null
