"""Make plots for the DLA dNdX estimation paper"""

import os.path as path
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import calc_cddf
import dla_data
from save_figure import save_figure

def do_data_plots(cat, subdir):
    """Make a set of plots"""
    dla_data.noterdaeme_12_data()
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(zmax=5,color="blue")
    np.savetxt(path.join(subdir,"cddf_all.txt"), (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]))
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_gp"))
    plt.clf()

    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(zmax=5,color="blue", moment=True)
    plt.xlim(1e20, 1e23)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_moment_gp"))
    plt.clf()

    #Evolution with redshift
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(4,5, label="4-5", color="brown")
    np.savetxt(path.join(subdir,"cddf_z45.txt"), (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]))
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(3,4, label="3-4", color="black")
    np.savetxt(path.join(subdir,"cddf_z34.txt"), (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]))
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(2.5,3, label="2.5-3", color="green")
    np.savetxt(path.join(subdir,"cddf_z253.txt"), (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]))
    (l_N, cddf, cddf68, cddf95) = cat.plot_cddf(2,2.5, label="2-2.5", color="blue")
    np.savetxt(path.join(subdir,"cddf_z225.txt"), (l_N, cddf, cddf68[:,0], cddf68[:,1], cddf95[:,0],cddf95[:,1]))
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"cddf_zz_gp"))
    plt.clf()

    #dNdX
    dla_data.dndx_not()
    dla_data.dndx_pro()
    (z_cent, dNdX, dndx68, dndx95) = cat.plot_line_density(zmax=5)
    np.savetxt(path.join(subdir,"dndx_all.txt"), (z_cent, dNdX, dndx68[:,0],dndx68[:,1], dndx95[:,0],dndx95[:,1]) )
    plt.legend(loc=0)
    plt.ylim(0,0.16)
    save_figure(path.join(subdir,"dndx_gp"))
    plt.clf()

    #Omega_DLA
    dla_data.omegahi_not()
    dla_data.omegahi_pro()
    dla_data.crighton_omega()
    (z_cent, omega_dla, omega_dla_68, omega_dla_95) = cat.plot_omega_dla(zmax=5)
#     cat.tophat_prior = True
#     cat.plot_omega_dla(zmax=5, label="Tophat Prior", twosigma=False)
#     cat.tophat_prior = False
    np.savetxt(path.join(subdir,"omega_dla_all.txt"), (z_cent, omega_dla, omega_dla_68[:,0],omega_dla_68[:,1], omega_dla_95[:,0], omega_dla_95[:,1]))
    plt.legend(loc=0)
    plt.xlim(2,5)
    plt.ylim(0,2.5)
    save_figure(path.join(subdir,"omega_gp"))
    plt.clf()

def do_sample_error_check(cat, subdir):
    """Do a bunch of resamplings to check the effect of sample variance."""
    #dNdX/Omega_DLA
    cat.plot_dndx_sample_errors(z_max=5,nsample=13)
    plt.legend(loc=0)
    plt.ylim(0,0.16)
    save_figure(path.join(subdir,"dndx_gp_resample"))
    plt.clf()
    cat.plot_omega_sample_errors(z_max=5,nsample=13)
    plt.legend(loc=0)
    plt.ylim(0,2.5)
    save_figure(path.join(subdir,"omega_gp_resample"))
    plt.clf()

def do_check_p_thresh(cat, subdir):
    """Check the effect of very unlikely samples"""
    cat.p_thresh_sample = 1e-4
    cat.plot_line_density(zmax=5, label=r"$p_\mathrm{sample} = 10^{-4}$")
    cat.p_thresh_sample = 1e-2
    cat.plot_line_density(zmax=5, label=r"$p_\mathrm{sample} = 10^{-2}$")
    cat.p_thresh_sample = 1e-4
    cat.p_thresh_spec = 0.1
    cat.plot_line_density(zmax=5, label=r"$p_\mathrm{spec} = 10^{-1}$")
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_p_thresh"))
    plt.clf()

def do_pixel_noise_check(cat, subdir):
    """Check effect of removing spectra with a low SNR."""
    cat.set_snr(1)
    nt = cat.noise_thresh
    cat.filter_noisy_pixels = True
    cat.plot_omega_dla(zmax=5,label="N < 0.5")
    cat.noise_thresh = 1.
    cat.plot_omega_dla(zmax=5,label="N < 1")
    cat.noise_thresh = 0.25**2
    cat.plot_omega_dla(zmax=5,label="N < 0.25")
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_gp_pix_noise"))
    plt.clf()

    cat.plot_line_density(zmax=5,label="N < 0.5")
    cat.noise_thresh = 1.
    cat.plot_line_density(zmax=5,label="N < 1")
    cat.noise_thresh = 0.25**2
    cat.plot_line_density(zmax=5,label="N < 0.25")
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_gp_pix_noise"))
    plt.clf()
    cat.noise_thresh = nt
    cat.filter_noisy_pixels = False

def do_snr_check(cat, subdir):
    """Check effect of removing spectra with a low SNR."""
    first_snr = cat.snr_thresh
    cat.set_snr(-2)
    cat.plot_omega_dla(zmax=5,label="All GP")
    cat.set_snr(2)
    cat.plot_omega_dla(zmax=5,label="SNR > 2")
    cat.set_snr(4)
    cat.plot_omega_dla(zmax=5,label="SNR > 4")
#     cat.set_snr(8)
#     cat.plot_omega_dla(zmax=5,label="SNR > 8")
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_gp_snr"))
    plt.clf()

    cat.set_snr(-2)
    cat.plot_line_density(zmax=5, label="All GP")
    cat.set_snr(2)
    cat.plot_line_density(zmax=5, label="SNR > 2")
    cat.set_snr(4)
    cat.plot_line_density(zmax=5, label="SNR > 4")
#     cat.set_snr(8)
#     cat.plot_line_density(zmax=5, label="SNR > 8")
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_gp_snr"))
    plt.clf()
    cat.set_snr(first_snr)

def do_lowzcut_check(cat, subdir):
    """Check effect of the low-z cut."""
    lowzcut = cat.lowzcut
    cat.lowzcut = True
    cat.plot_omega_dla(zmax=5,label="Cutting")
    cat.lowzcut = False
    cat.plot_omega_dla(zmax=5,label="Not cutting")
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_gp_lowz"))
    plt.clf()

    cat.lowzcut = True
    cat.plot_line_density(zmax=5,label="Cutting")
    cat.lowzcut = False
    cat.plot_line_density(zmax=5,label="Not cutting")
    plt.ylim(0,0.12)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_gp_lowz"))
    plt.clf()
    cat.lowzcut = lowzcut

def do_2dla_plots(cat, subdir):
    """Check the effect of a second DLA. No longer included in catalogue"""
    #Omega_DLA in variance vs bayesian mode
    cat.second_dla=False
    cat.plot_omega_dla(zmax=5, label="Confidence interval")
    cat.second_dla=True
    cat.plot_omega_dla_var(zmax=5, label="Variance")
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_gp_diff"))
    plt.clf()

    #dNdX
    #Check effect of the second DLA
    cat.plot_line_density(zmax=5,label="Two-DLA")
    cat.second_dla = False
    cat.plot_line_density(zmax=5,label="One-DLA")
    cat.second_dla = True
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_2dla"))
    plt.clf()

    cat.plot_omega_dla(zmax=5,label="Two-DLA")
    cat.second_dla = False
    cat.plot_omega_dla(zmax=5,label="One-DLA")
    cat.second_dla = True
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_2dla"))
    plt.clf()

def do_qso_split(cat, subdir):
    """Check the effect of the quasar redshift."""
    #Check z_qso split
    oldcond = cat.condition
    high_z = (2.5,3.0,3.5,5.0)
    low_z = (2.0,2.5,3.0,3.5)
    for (high_z_qso, z_qso_split) in zip(high_z, low_z):
        cat.condition = (cat.z_max() < high_z_qso)*(cat.z_max() > z_qso_split)
        cat.plot_omega_dla(label="$"+str(high_z_qso)+" > z_\mathrm{QSO} > "+str(z_qso_split)+"$")
    plt.ylim(ymin=0)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_gp_zqso"+str(cat.lowzcut)))
    plt.clf()

    for (high_z_qso, z_qso_split) in zip(high_z, low_z):
        cat.condition = (cat.z_max() < high_z_qso)*(cat.z_max() > z_qso_split)
        cat.plot_line_density(label="$"+str(high_z_qso)+" > z_\mathrm{QSO} > "+str(z_qso_split)+"$")
    plt.ylim(ymin=0,ymax=0.15)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_gp_zqso"+str(cat.lowzcut)))
    plt.clf()
    cat.condition = oldcond

def do_length_split(cat, subdir):
    """Check the effect of the quasar redshift."""
    #Check z_qso split
    oldcond = cat.condition
    high_z = (0.2,0.4,0.6,0.8,2)
    low_z = (0., 0.2, 0.4, 0.6, 0.8)
    z_diff = cat.z_max() - cat.z_min()
    for (high_z_qso, z_qso_split) in zip(high_z, low_z):
        cat.condition = (z_diff < high_z_qso)*(z_diff > z_qso_split)
        cat.plot_omega_dla(label=str(high_z_qso)+" > zQSO > "+str(z_qso_split))
    plt.ylim(ymin=0)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_gp_zdiff"))
    plt.clf()

    for (high_z_qso, z_qso_split) in zip(high_z, low_z):
        cat.condition = (z_diff < high_z_qso)*(z_diff > z_qso_split)
        cat.plot_line_density(label=str(high_z_qso)+" > zQSO > "+str(z_qso_split))
    plt.ylim(ymin=0,ymax=0.1)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_gp_zdiff"))
    plt.clf()
    cat.condition = oldcond

def do_compare_plots(cat7, cat7s, subdir,label):
    """Plots to compare two cddfs"""
    #Check the effect of the 5km/s split
    #dNdX
    cat7.plot_line_density(zmax=5)
    cat7s.plot_line_density(zmax=5, label=label)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"dndx_"+label))
    plt.clf()

    #Omega_DLA
    cat7.plot_cddf(zmax=4,color="blue")
    cat7s.plot_cddf(zmax=4,color="red",label=label)
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(path.join(subdir, "cddf_"+label))
    plt.clf()

    #Omega_DLA
    cat7.plot_omega_dla(zmax=5)
    cat7s.plot_omega_dla(zmax=5, label=label)
    plt.legend(loc=0)
    save_figure(path.join(subdir,"omega_"+label))
    plt.clf()

if __name__=="__main__":
    #DR7 data
    #Using old samples
    #cat7ss = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q.mat", snr=-2)
    #do_snr_check(cat7ss, "DR7")

    #cat7 = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q.mat")
    #do_data_plots(cat7,"DR7")
    #print("Done data plots")
    #do_check_plots(cat7,"DR7")
    #print("Done check plots")
    #cat7p = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q.mat")
    #do_check_p_thresh(cat7p, "DR7")
    #del cat7p
    #print("Done p_thresh")
    #do_pixel_noise_check(cat7ss, "DR7")
    #del cat7ss
    #print("Done SNR")

    #cat7s = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr7q_5kms_separation.mat")

    #do_compare_plots(cat7,cat7s,"DR7", label="5kms")

    #DR12 data
    cat12 = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr12q_lyb_lya.mat", snrs_file = "snrs_qsos_dr12.mat")
    do_data_plots(cat12,"DR12")
    cat12.lowzcut=False
    do_qso_split(cat12,"DR12")
    cat12.lowzcut=True
    do_qso_split(cat12,"DR12")
    cat12.lowzcut=False
#     do_length_split(cat12, "DR12")
    do_lowzcut_check(cat12, "DR12")
    do_snr_check(cat12, "DR12")
    do_sample_error_check(cat12, "DR12")
    #do_pixel_noise_check(cat12, "DR12")
    #cat12p = calc_cddf.DLACatalogue(processed_file="processed_qsos_dr12q.mat", snrs_file = "snrs_qsos_dr12.mat")
    #do_check_p_thresh(cat12p, "DR12")

