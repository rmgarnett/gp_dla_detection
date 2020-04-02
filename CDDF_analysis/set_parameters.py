'''
set_paramters.py : set gp_dla_detection default parameters for cddf module
'''

# physical constants
lya_wavelength = 1215.6701
lyb_wavelength = 1025.7223
lyman_limit    = 911.7633
speed_of_light = 299792458

# oscillator strengths
lya_oscillator_strength = 0.416400
lyb_oscillator_strength = 0.079120

# all transition wavelengths 
all_transition_wavelengths = [
    1.2156701e-05,
    1.0257223e-05,
    9.725368e-06,
    9.497431e-06,
    9.378035e-06,
    9.307483e-06,
    9.262257e-06,
    9.231504e-06,
    9.209631e-06,
    9.193514e-06,
    9.181294e-06,
    9.171806e-06,
    9.16429e-06,
    9.15824e-06,
    9.15329e-06,
    9.14919e-06,
    9.14576e-06,
    9.14286e-06,
    9.14039e-06,
    9.13826e-06,
    9.13641e-06,
    9.13480e-06,
    9.13339e-06,
    9.13215e-06,
    9.13104e-06,
    9.13006e-06,
    9.12918e-06,
    9.12839e-06,
    9.12768e-06,
    9.12703e-06,
    9.12645e-06
] # in cgs
all_transition_wavelengths = [
    l * 10**8 for l in all_transition_wavelengths
] # in angstrom

all_oscillator_strengths = [
    0.416400,
    0.079120,
    0.029000,
    0.013940,
    0.007799,
    0.004814,
    0.003183,
    0.002216,
    0.001605,
    0.00120,
    0.000921,
    0.0007226,
    0.000577,
    0.000469,
    0.000386,
    0.000321,
    0.000270,
    0.000230,
    0.000197,
    0.000170,
    0.000148,
    0.000129,
    0.000114,
    0.000101,
    0.000089,
    0.000080,
    0.000071,
    0.000064,
    0.000058,
    0.000053,
    0.000048
]

# convert relative velocity to redshift
kms_to_z = lambda kms : kms * 1000 / speed_of_light

# utility functions for redshifting
emitted_wavelengths  = lambda observed_wavelengths, z : observed_wavelengths / (1 + z)
observed_wavelengths = lambda emitted_wavelengths, z  : emitted_wavelengths  * (1 + z) 

# base directory for all data
base_directory = 'data'

# utility functions for directories
processed_directory = lambda release : "{}/{}/processed".format(base_directory, release)
spectra_directory   = lambda release : "{}/{}/spectra".format(base_directory, release)
