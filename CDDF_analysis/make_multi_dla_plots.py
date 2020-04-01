'''
Make plots for the Multi-DLA paper
'''
import os
import numpy as np 
from scipy.interpolate import interp1d
import matplotlib 
from matplotlib import pyplot as plt 
from matplotlib import cm
from .qso_loader import QSOLoader, file_loader
from .set_parameters import *
from .dla_data import dla_data

cmap = cm.get_cmap('viridis')

# change fontsize
matplotlib.rcParams.update({'font.size' : 14})

# matplotlib.use('PDF')

save_figure = lambda filename : plt.savefig("{}.pdf".format(filename), format="pdf", dpi=300)


def generate_qsos(base_directory="", release="dr12q",
        dla_concordance="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        los_concordance="data/dla_catalogs/dr9q_concordance/processed/los_catalog"):
    '''
    Return a pair of QSOLoader instances : multi-dla, original
    '''
    preloaded_file = os.path.join( 
        base_directory, processed_directory(release), "preloaded_qsos.mat")
    processed_file_multidla  = os.path.join(
        base_directory, processed_directory(release), "processed_qsos_multi_lyseries_a03_zwarn_occams_trunc_dr12q.mat" )
    processed_file_original = os.path.join(
        base_directory, processed_directory(release), "processed_qsos_dr12q.mat" )
    catalogue_file = os.path.join(
        base_directory, processed_directory(release), "catalog.mat")
    learned_file_multidla   = os.path.join(
        base_directory, processed_directory(release), "learned_qso_model_lyseries_variance_kim_dr9q_minus_concordance.mat")
    learned_file_original   = os.path.join(
        base_directory, processed_directory(release), "learned_qso_model_dr9q_minus_concordance.mat")
    snrs_file_multidla      = os.path.join(
        base_directory, processed_directory(release), "snrs_qsos_multi_dr12q_zwarn.mat")
    snrs_file_original      = os.path.join(
        base_directory, processed_directory(release), "snrs_qsos_dr12q.mat")

    sample_file_original    = os.path.join(
        base_directory, processed_directory(release), "dla_samples.mat")

    qsos_multidla = QSOLoader(
        preloaded_file, catalogue_file, learned_file_multidla, processed_file_multidla, dla_concordance, los_concordance, snrs_file_multidla, 
        occams_razor=1)
    qsos_original = QSOLoader(
        preloaded_file, catalogue_file, learned_file_original, processed_file_original, dla_concordance, los_concordance, snrs_file_original, 
        sub_dla=False, sample_file=sample_file_original, occams_razor=1)

    return qsos_multidla, qsos_original

def generate_qsos_lyseries(base_directory="", release="dr12q",
        dla_concordance="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        los_concordance="data/dla_catalogs/dr9q_concordance/processed/los_catalog"):
    '''
    Return a pair of QSOLoader instances : lyman series, lyman alpha
    '''
    preloaded_file = os.path.join( 
        base_directory, processed_directory(release), "preloaded_qsos.mat")
    processed_file_lyseries  = os.path.join(
        base_directory, processed_directory(release), "processed_qsos_multi_lls_occam_dr12q.mat" )
    processed_file_lya1pz    = os.path.join(
        base_directory, processed_directory(release), "processed_qsos_multi_lls_occam_dr12q.mat" )
    catalogue_file = os.path.join(
        base_directory, processed_directory(release), "catalog.mat")
    learned_file_lyseries   = os.path.join(
        base_directory, processed_directory(release), "learned_qso_model_lyseries_kim_dr9q_minus_concordance.mat")
    learned_file_lyap1z   = os.path.join(
        base_directory, processed_directory(release), "learned_qso_model_lya1pz_kim_dr9q_minus_concordance.mat")
    snrs_file      = os.path.join(
        base_directory, processed_directory(release), "snrs_qsos_multi_dr12q.mat")

    qsos_lyseries = QSOLoader(
        preloaded_file, catalogue_file, learned_file_lyseries, processed_file_lyseries, dla_concordance, los_concordance, snrs_file)
    qsos_lya1pz   = QSOLoader(
        preloaded_file, catalogue_file, learned_file_lyap1z, processed_file_lya1pz, dla_concordance, los_concordance, snrs_file)

    return qsos_lyseries, qsos_lya1pz

def do_procedure_plots(qsos_multidla, qsos_original):
    # scaling factor between rest_wavelengths to pixels
    min_lambda = lyman_limit - 10
    max_lambda = lya_wavelength + 10
    scale = 1 / ( max_lambda - min_lambda )

    # compare different learned mus
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    # ax[0].plot(
    #     qsos_multidla.GP.rest_wavelengths, qsos_multidla.GP.mu, label=r"re-trained $\mu$")
    # ax[0].plot(
    #     qsos_original.GP.rest_wavelengths, qsos_original.GP.mu, label=r"Garnett (2017) $\mu$", color="lightblue")

    # ax[0].legend()
    # ax[0].set_xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $(\AA)$")
    # ax[0].set_ylabel(r"normalised flux")
    # ax[0].set_xlim([min_lambda, max_lambda])

    # ax02 = ax[0].twiny()
    # ax02.set_xticks(
    #     [(lyman_limit - min_lambda) * scale, (lyb_wavelength - min_lambda) * scale, (lya_wavelength - min_lambda) * scale])
    # ax02.set_xticklabels([r"Ly $\infty$", r"Ly $\beta$", r"Ly $\alpha$"])

    # compare different learned omegas
    ax.plot(
        qsos_multidla.GP.rest_wavelengths, np.exp(qsos_multidla.GP.log_omega), label=r"re-trained $\omega$")
    ax.plot(
        qsos_original.GP.rest_wavelengths, np.exp( qsos_original.GP.log_omega ), label=r"Garnett (2017) $\omega$", color="lightblue")

    ax.legend()
    ax.set_xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
    ax.set_ylabel(r"normalised flux")
    ax.set_xlim([min_lambda, max_lambda])

    ax12 = ax.twiny()
    ax12.set_xticks(
        [(lyman_limit - min_lambda) * scale, (lyb_wavelength - min_lambda) * scale, (lya_wavelength - min_lambda) * scale])
    ax12.set_xticklabels([r"Ly $\infty$", r"Ly $\beta$", r"Ly $\alpha$"])

    plt.tight_layout()
    save_figure("mu_omega_changes")
    plt.clf()

    # plotting covariance matrix
    min_lambda = lyman_limit 
    max_lambda = lya_wavelength
    scale = np.shape(qsos_multidla.GP.C)[0] / ( max_lambda - min_lambda )

    lyg_wavelength = 972

    fix, ax = plt.subplots(figsize=(6,6))
    ax.imshow(qsos_multidla.GP.C, origin="lower")
    ax.set_xticks(
        [(lyman_limit - min_lambda) * scale, (lyg_wavelength - min_lambda) * scale,
        (lyb_wavelength - min_lambda) * scale, (lya_wavelength - min_lambda) * scale])
    ax.set_xticklabels([r"Ly $\infty$", r"Ly $\gamma$", r"Ly $\beta$", r"Ly $\alpha$"])
    ax.set_yticks(
        [(lyman_limit - min_lambda) * scale, (lyg_wavelength - min_lambda) * scale,
        (lyb_wavelength - min_lambda) * scale, (lya_wavelength - min_lambda) * scale])
    ax.set_yticklabels([r"Ly $\infty$", r"Ly $\gamma$", r"Ly $\beta$", r"Ly $\alpha$"])

    plt.tight_layout()
    save_figure("covariance_matrix")
    plt.clf()

def do_meanflux_samples(qsos_multidla, qsos_original):
    '''
    Plot some examples to demonstrate the mean-flux suppression
    
    Examples (z_qso > 5)
    ---
    nspec = 156180; z_qso = 5.068
    '''
    nspec = 72

    qsos_multidla.plot_mean_flux(nspec)
    plt.plot(qsos_multidla.GP.rest_wavelengths, qsos_multidla.GP.mu, label=r"$\mu$, before suppression", color="red", ls=':')
    plt.ylim(-1, 8)
    plt.legend()

    plt.tight_layout()
    save_figure("meanflux_{}".format(nspec))
    plt.clf()

def do_this_mu_examples(qsos):
    '''
    Plot some examples to compare the detections between Parks and Multi-DLA model.
    '''
    specs = [134315, ]

    for spec in specs:
        qsos.plot_this_mu(spec, Parks=True, dla_parks='multi_dlas/predictions_DR12.json', num_forest_lines=31)
        plt.ylim(-1, 8)
        save_figure('this_mu_{}'.format(spec))

def do_lyman_series_suppression(qsos_lyseries, qsos_lya1pz, n_spec = 156180):
    '''
    Plot the difference of considering lyman series
    '''    
    this_wavelengths = qsos_lyseries.find_this_wavelengths(n_spec) / (1 + qsos_lyseries.z_qsos[n_spec])
    this_flux        = qsos_lyseries.find_this_flux(n_spec)

    assert np.all( (this_flux - qsos_lya1pz.find_this_flux(n_spec)) < 0.1)

    rest_wavelengths, this_mu_a = qsos_lyseries.plot_mean_flux(n_spec, suppressed=True, num_lines=31)
    plt.clf()
    plt.close()
    rest_wavelengths, this_mu_b = qsos_lya1pz.plot_mean_flux(n_spec,   suppressed=True, num_lines=1)
    plt.clf()
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(this_wavelengths, this_flux, label=r"spec-{}-{}-{}, z_qso = {:.3g}".format(
        qsos_lyseries.plates[n_spec], qsos_lyseries.mjds[n_spec], qsos_lyseries.fiber_ids[n_spec], qsos_lyseries.z_qsos[n_spec]))
    plt.plot(rest_wavelengths, this_mu_a, label=r"num_lines = 31", color="red")
    plt.plot(rest_wavelengths, this_mu_b, label=r"num_lines =  1", color="red", ls=':')
    plt.ylim(-1, 8)
    plt.legend()

    plt.tight_layout()
    save_figure("test_num_lines_{}".format(n_spec))
    plt.clf()

def do_MAP_concordance_comparison(qsos, num_bins=100):
    '''
    do the MAP comparison with DR9 concorndace catalogue
    '''
    Delta_z_dlas, Delta_log_nhis = qsos.make_MAP_comparison(qsos.dla_catalog)

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    axs[0].hist(Delta_z_dlas, bins=np.linspace(-0.01, 0.01, num_bins), density=True)
    axs[0].set_xlim(-0.01, 0.01) # this is set to be identical to Garnett (2017)
    axs[0].set_xlabel(r'difference between $z_{DLA}$ (new code/concordance)')
    axs[0].set_ylabel(r'$p$(difference)')

    axs[1].hist(Delta_log_nhis, bins=np.linspace(-1, 1, num_bins), density=True)
    axs[1].set_xlim(-1, 1)
    axs[1].set_xlabel(r'difference between $\log{N_{HI}}$ (new code/concordance)')
    axs[1].set_ylabel(r'$p$(difference)')
    
    plt.tight_layout()
    save_figure('MAP_comparison_concordance')
    plt.clf()
    plt.close()

def do_MAP_parks_comparison(qsos, dla_parks, dla_confidence=0.98, num_dlas=1, num_bins=100):
    '''
    Plot the comparisons between MAP values and Parks' predictions
    '''
    if 'dla_catalog_parks' not in dir(qsos):
        qsos.load_dla_parks(dla_parks, p_thresh=dla_confidence, multi_dla=False, num_dla=1)

    Delta_z_dlas, Delta_log_nhis, z_dlas_parks = qsos.make_MAP_parks_comparison(
        qsos.dla_catalog_parks, num_dlas=num_dlas, dla_confidence=dla_confidence)

    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    # plot in the same scale as in the Garnett (2017) paper
    # for z_dlas, xlim(-0.01, 0.01); for log_nhis, xlim(-1, 1)
    # TODO: make KDE plot
    axs[0].hist(Delta_z_dlas.ravel(), 
        bins=np.linspace(-0.01, 0.01, num_bins), density=True)
    axs[0].set_xlim(-0.01, 0.01)
    axs[0].set_xlabel(r'difference between $z_{DLA}$ (new code/Parks);'+' DLA({})'.format(num_dlas))
    axs[0].set_ylabel(r'$p$(difference)')

    axs[1].hist(Delta_log_nhis.ravel(), 
        bins=np.linspace(-1, 1, num_bins), density=True)
    axs[1].set_xlim(-1, 1)
    axs[1].set_xlabel(r'difference between $\log{N_{HI}}$ (new code/Parks);'+' DLA({})'.format(num_dlas))
    axs[1].set_ylabel(r'$p$(difference)')

    plt.tight_layout()
    save_figure('MAP_comparison_Parks_dlas{}_confidence{}'.format(num_dlas, str(dla_confidence).replace(".","_")))
    plt.clf()
    plt.close()   

    high_z = np.array([2.5,3.0,3.5,5.0])
    low_z = np.array([2.0,2.5,3.0,3.5])
    log_nhi_evo = []; sterr_log_nhi_evo = []
    z_dla_evo   = []; sterr_z_dla_evo   =  []
    for high_z_dla, low_z_dla in zip(high_z, low_z):
        inds = (z_dlas_parks > low_z_dla) & (z_dlas_parks < high_z_dla)

        z_dla_evo.append( np.nanmean( Delta_z_dlas[inds] ) )
        log_nhi_evo.append( np.nanmean( Delta_log_nhis[inds] ) )

        # stderr = sqrt(s / n)
        sterr_z_dla_evo.append( 
            np.sqrt( np.nansum((Delta_z_dlas[inds] - np.nanmean(Delta_z_dlas[inds]))**2 ) / (Delta_z_dlas[inds].shape[0] - 1)) 
            / np.sqrt(Delta_z_dlas[inds].shape[0]) )

        sterr_log_nhi_evo.append( 
            np.sqrt( np.nansum( (Delta_log_nhis[inds] - np.nanmean(Delta_log_nhis[inds]))**2 ) / (Delta_log_nhis[inds].shape[0] - 1)) 
            / np.sqrt(Delta_log_nhis[inds].shape[0]) )

    z_cent = np.array( [(z_x + z_m) / 2. for (z_m, z_x) in zip(high_z, low_z)] )
    xerrs  = (z_cent - low_z, high_z - z_cent)

    plt.errorbar(z_cent, z_dla_evo, yerr=sterr_z_dla_evo, xerr=xerrs, 
        label=r"$z_{{DLA}} - z_{{DLA}}^{{Parks}} \mid \mathrm{{Garnett}}({k}) \cap \mathrm{{Parks}}({k})$".format(k=num_dlas))
    plt.xlabel(r"$z_{DLA}^{Parks}$")
    plt.ylabel(r"$\Delta z_{DLA}$")
    plt.legend(loc=0)
    save_figure("MAP_z_dla_evolution_parks")
    plt.clf()

    plt.errorbar(z_cent, log_nhi_evo, yerr=sterr_log_nhi_evo, xerr=xerrs, 
        label=r"$\log{{N_{{HI}}}}_{{DLA}} - \log{{N_{{HI}}}}_{{DLA}}^{{Parks}} \mid \mathrm{{Garnett}}({k}) \cap \mathrm{{Parks}}({k})$".format(k=num_dlas))
    plt.xlabel(r"$z_{DLA}^{Parks}$")
    plt.ylabel(r"$\Delta \log{N_{HI}}$")
    plt.legend(loc=0)
    save_figure("MAP_log_nhi_evolution_parks")
    plt.clf()

def do_MAP_Garnett_comparison(qsos_multidla, qsos_original):
    # map values array size: (num_qsos, model_DLA(n), num_dlas)
    inds = np.isin( qsos_original.thing_ids, qsos_multidla.thing_ids )

    # get rid of thing_id == -1
    inds = inds & (qsos_original.thing_ids != -1)
    inds_multi = qsos_multidla.thing_ids != -1

    # qsos_original.prepare_roman_map_vals(sample_file="data/dr12q/processed/dla_samples.mat", use_memory=True, split=20)

    # query the DLA index from Garnett and current model
    dla_inds = qsos_original.p_dlas[inds] > 0.5
    dla_inds = dla_inds & (qsos_multidla.dla_map_model_index[inds_multi] > qsos_multidla.sub_dla)

    Delta_z_dlas   = qsos_multidla.map_z_dlas[inds_multi, 0, 0][dla_inds] - qsos_original.all_z_dlas[inds][dla_inds]
    Delta_log_nhis = qsos_multidla.map_log_nhis[inds_multi, 0, 0][dla_inds] - qsos_original.all_log_nhis[inds][dla_inds]

    return Delta_z_dlas, Delta_log_nhis

def do_confusion_parks(qsos, dla_parks, snr=-1, dla_confidence=0.98, p_thresh=0.98, lyb=False):
    '''
    plot the multi-DLA confusion matrix between our MAP predictions and Parks' predictions 
    '''
    if 'dla_catalog_parks' not in dir(qsos):
        qsos.load_dla_parks(dla_parks, p_thresh=dla_confidence, multi_dla=False, num_dla=1)

    confusion_matrix,_ = qsos.make_multi_confusion(qsos.dla_catalog_parks, dla_confidence, p_thresh, snr=snr, lyb=lyb)

    size, _ = confusion_matrix.shape

    print("Confusion Matrix Garnett's Multi-DLA versus Parks : ")
    print("----")
    for i in range(size):
        print("{} DLA".format(i), end="\t")
        for j in range(size):
            print("{}".format(confusion_matrix[i,j]), end=" ")   
        print("")

    print("Mutli-DLA disagreements : ")
    print("----")
    for i in range(size):
        num = confusion_matrix[(i+1):, 0:(i+1)].sum() + confusion_matrix[0:(i+1), (i+1):].sum()
        print("Error between >= {} DLAs and < {} DLAs: {:.2g}".format(i + 1, i + 1, num / confusion_matrix.sum()))


def do_ROC_comparisons(qsos_multidla, qsos_original, occams_razor=10000):
    '''
    Plot Two ROC curves to demonstrate the difference

    Parameters:
    ----
    occams_razor : N > 0, only for multi-DLA
    '''
    TPR_multi, FPR_multi   = qsos_multidla.make_ROC(qsos_multidla.dla_catalog, occams_razor=occams_razor)
    TPR_origin, FPR_origin = qsos_original.make_ROC(qsos_original.dla_catalog, occams_razor=occams_razor)
    
    from scipy.integrate import cumtrapz

    AUC_multi  = - cumtrapz(TPR_multi, x=FPR_multi)[-1]
    AUC_origin = - cumtrapz(TPR_origin, x=FPR_origin)[-1] 

    plt.plot(FPR_multi,  TPR_multi,  color="C1",        label="current;  AUC: {:.3g}".format(AUC_multi))
    plt.plot(FPR_origin, TPR_origin, color="lightblue", label="original; AUC: {:.3g}".format(AUC_origin))
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.legend()
    save_figure("ROC_multi")
    plt.clf()

def do_multi_ROC(qsos, dla_parks):
    '''
    Plot the ROC curve for multi-DLAs, comparing to Parks (2018)
    '''
    qsos.load_dla_parks(dla_parks, p_thresh=0.98, multi_dla=True, num_dla=2)
    
    TPR, FPR = qsos.make_multi_ROC(qsos.dla_catalog_parks)

    from scipy.integrate import cumtrapz

    AUC = - cumtrapz(TPR, x=FPR)[-1]

    plt.plot(FPR, TPR, color="C1", label="AUC: {:.3g}".format(AUC))
    plt.title(r"$\mathcal{M}_{DLA(1 + 2)}$ comparing to Parks(2018)")
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.legend()
    save_figure("ROC_multi_parks")
    plt.clf()

def do_NoterdaemeDR12_CDDF(qsos, dla_noterdaeme, los_noterdaeme, subdir, zmax=3.5, snr_thresh=4):
    '''
    Plot the column density distribution function of Noterdaeme DR12
    '''
    dla_data.noterdaeme_12_data()
    (l_N, cddf) = qsos.plot_cddf_noterdaeme(
        dla_noterdaeme, los_noterdaeme, zmax=zmax, snr_thresh=snr_thresh, color="blue")
    np.savetxt(
        os.path.join(subdir, "cddf_noterdaeme_all.txt"),
        (l_N, cddf))
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_noterdaeme"))
    plt.clf()

    # Evolution with redshift
    (l_N, cddf) = qsos.plot_cddf_noterdaeme(
        dla_noterdaeme, los_noterdaeme, zmin=4, zmax=5, label="4-5", color="brown")
    np.savetxt(
        os.path.join(subdir, "cddf_noterdaeme_z45.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_noterdaeme(
        dla_noterdaeme, los_noterdaeme, zmin=3, zmax=4, label="3-4", color="black")
    np.savetxt(
        os.path.join(subdir, "cddf_noterdaeme_z34.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_noterdaeme(
        dla_noterdaeme, los_noterdaeme, zmin=2.5, zmax=3, label="2.5-3", color="green")
    np.savetxt(
        os.path.join(subdir, "cddf_noterdaeme_z253.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_noterdaeme(
        dla_noterdaeme, los_noterdaeme, zmin=2, zmax=2.5, label="2-2.5", color="blue")
    np.savetxt(
        os.path.join(subdir, "cddf_noterdaeme_z225.txt"), (l_N, cddf))

    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_zz_noterdaeme"))
    plt.clf()    

def do_NoterdaemeDR12_dNdX(qsos, dla_noterdaeme, los_noterdaeme, subdir, zmax=3.5, snr_thresh=4):
    '''
    Plot line density of Noterdaeme DR12
    '''
    dla_data.dndx_not()
    dla_data.dndx_pro()
    z_cent, dNdX = qsos.plot_line_density_noterdaeme(
        dla_noterdaeme, los_noterdaeme, zmax=zmax, snr_thresh=snr_thresh)
    np.savetxt(os.path.join(subdir, "dndx_noterdaeme_all.txt"), (z_cent, dNdX))

    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    save_figure(os.path.join(subdir, "dndx_noterdaeme"))
    plt.clf()


def do_Parks_CDDF(qsos, dla_parks, subdir, p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
    '''
    Plot the column density function of Parks (2018)

    Parameters:
    ----
    dla_parks (str) : path to Parks' `prediction_DR12.json`
    '''
    dla_data.noterdaeme_12_data()
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_parks, zmax=5, color="blue", p_thresh=p_thresh, snr_thresh=snr_thresh, prior=False, apply_p_dlas=False)
    np.savetxt(
        os.path.join(subdir, "cddf_parks_all.txt"),
        (l_N, cddf))
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_parks"))
    plt.clf()

    # Evolution with redshift
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_parks, zmin=4, zmax=5, label="4-5", color="brown", 
        p_thresh=p_thresh, snr_thresh=snr_thresh, prior=False, apply_p_dlas=False)
    np.savetxt(
        os.path.join(subdir, "cddf_parks_z45.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_parks, zmin=3, zmax=4, label="3-4", color="black", 
        p_thresh=p_thresh, snr_thresh=snr_thresh, prior=False, apply_p_dlas=False)
    np.savetxt(
        os.path.join(subdir, "cddf_parks_z34.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_parks, zmin=2.5, zmax=3, label="2.5-3", color="green", 
        p_thresh=p_thresh, snr_thresh=snr_thresh, prior=False, apply_p_dlas=False)
    np.savetxt(
        os.path.join(subdir, "cddf_parks_z253.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf_parks(
        dla_parks, zmin=2, zmax=2.5, label="2-2.5", color="blue", 
        p_thresh=p_thresh, snr_thresh=snr_thresh, prior=False, apply_p_dlas=False)
    np.savetxt(
        os.path.join(subdir, "cddf_parks_z225.txt"), (l_N, cddf))

    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_zz_parks"))
    plt.clf()

def do_Parks_dNdX(qsos, dla_parks, subdir, p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
    '''
    Plot dNdX for Parks' (2018)

    Parameters:
    ----
    dla_parks (str) : path to Parks' `prediction_DR12.json`
    '''
    dla_data.dndx_not()
    dla_data.dndx_pro()
    z_cent, dNdX = qsos.plot_line_density_park(
        dla_parks, zmax=5, p_thresh=p_thresh, snr_thresh=snr_thresh, prior=False, apply_p_dlas=False)
    np.savetxt(os.path.join(subdir, "dndx_all.txt"), (z_cent, dNdX))

    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    save_figure(os.path.join(subdir, "dndx_parks"))
    plt.clf()

def do_Parks_snr_check(qsos, dla_parks, subdir, p_thresh=0.98, prior=False, apply_p_dlas=False):
    '''
    Check effect of removing spectra with low SNRs.
    '''
    snrs_list = (-2, 2, 4, 8)

    # CDDF
    dla_data.noterdaeme_12_data()
    for i,snr_thresh in enumerate(snrs_list):
        (l_N, cddf) = qsos.plot_cddf_parks(
            dla_parks, zmax=5, p_thresh=p_thresh, color=cmap( (i + 1)  / len(snrs_list)),
            snr_thresh=snr_thresh, label="Parks SNR > {:d}".format(snr_thresh), 
            prior=prior, apply_p_dlas=apply_p_dlas)

    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_parks_snr"))
    plt.clf()
    
    # dN/dX
    dla_data.dndx_not()
    dla_data.dndx_pro()
    for i,snr_thresh in enumerate(snrs_list):
        z_cent, dNdX = qsos.plot_line_density_park(
            dla_parks, zmax=5, p_thresh=p_thresh, color=cmap( (i + 1)  / len(snrs_list)),
            snr_thresh=snr_thresh, label="Parks SNR > {:d}".format(snr_thresh), 
            prior=prior, apply_p_dlas=apply_p_dlas)

    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    save_figure(os.path.join(subdir, "dndx_parks_snr"))
    plt.clf()

def do_NoterdaemeDR12_snr_check(qsos, dla_noterdaeme, los_noterdaeme, subdir, zmax=3.5):
    '''
    Check effect of removing spectra with low SNRs.
    '''
    snrs_list = (-2, 2, 4, 8, 16)

    # CDDF
    dla_data.noterdaeme_12_data()
    for i,snr_thresh in enumerate(snrs_list):
        (l_N, cddf) = qsos.plot_cddf_noterdaeme(
            dla_noterdaeme, los_noterdaeme, zmax=zmax, 
            color=cmap( (i + 1) / len(snrs_list) ), 
            snr_thresh=snr_thresh, label="N DR12 SNR > {:d}".format(snr_thresh))
    
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_noterdaeme_snr"))
    plt.clf()

    # dN/dX
    dla_data.dndx_not()
    dla_data.dndx_pro()
    for i,snr_thresh in enumerate(snrs_list):
        z_cent, dNdX = qsos.plot_line_density_noterdaeme(
            dla_noterdaeme, los_noterdaeme, zmax=zmax, 
            color=cmap( (i + 1) / len(snrs_list) ), 
            snr_thresh=snr_thresh, label="N DR12 SNR > {:d}".format(snr_thresh))

    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    plt.xlim(2, 5)
    save_figure(os.path.join(subdir, "dndx_parks_snr"))
    plt.clf()


def do_Lya_demo(qsos, zmin=2, zmax=6, nbins=9, num_spec_bin=1, release='dr12q', dlambda=2.5):
    '''
    Do the Lyman alpha forest demo with fixed normalizer at redward of Lya
    '''
    zbins  = np.linspace(zmin, zmax, num=nbins+1)
    zcents = [(z1 + z2) / 2  for z1,z2 in zip(zbins[:-1], zbins[1:])] 

    nspec_list = [] # size(nbins, num_spec_bin)

    for z1, z2 in zip(zbins[:-1], zbins[1:]):
        nspecs = np.where( (qsos.z_qsos > z1) & (qsos.z_qsos < z2) )[0]
        
        # randomly choose samples from the selection of nspecs
        np.random.seed(1)
        choices = np.random.choice(nspecs, size=num_spec_bin)

        nspec_list.append(choices)

    # plot specs r.s.t redshift bins
    plt.figure(figsize=(16, 5))

    for i,nspecs in enumerate(nspec_list):
        z = zcents[i]

        for nspec in nspecs:
            # download files
            filename = file_loader(
                release, qsos.plates[nspec], qsos.mjds[nspec], qsos.fiber_ids[nspec])

            # check if the file already existed
            if not os.path.exists(filename):
                dirname = "{}/{:d}".format(
                        spectra_directory(release), qsos.plates[nspec])
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)

                # download raw data from sdss website
                qsos.retrieve_raw_spec(
                    qsos.plates[nspec], qsos.mjds[nspec], qsos.fiber_ids[nspec], release=release)

            # read fits file
            wavelengths, flux, noise_variance, pixel_mask = qsos.read_spec(filename)

            this_rest_wavelengths = emitted_wavelengths(wavelengths, qsos.z_qsos[nspec])

            # normalizer at the range of redward to the Lya
            inds = (this_rest_wavelengths >= normalization_min_lambda) & \
                   (this_rest_wavelengths <= normalization_max_lambda) & \
                   (~ pixel_mask)

            this_median = np.nanmedian( flux[inds] )

            flux           = flux / this_median
            noise_variance = noise_variance / this_median**2

            # interpolate with 0.25 A
            f = interp1d(this_rest_wavelengths, flux)
            rest_wavelengths = np.arange(
                this_rest_wavelengths.min(), this_rest_wavelengths.max(), step=dlambda)
            flux_smoothed = f(rest_wavelengths)

            # plot the full spec in rest-frame
            plt.plot(
                rest_wavelengths, flux_smoothed, color=cmap( (i + 1) / len(zcents) ), lw=1.5, label="zcent={:.3g}".format(z), alpha=0.8)

    plt.xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $(\AA)$")
    plt.ylabel(r"normalised flux")

    plt.ylim(-1, 8)
    plt.legend()
    save_figure("Lya_forest_demo")
    plt.clf()

def check_skylines(qsos):
    min_z_seperation = 0.01
    min_flux_thresh = 8

    skyline_dlas = []

    for nspec,z_qso in enumerate(qsos.z_qsos):
        print("inspecting {}/{} spec ... ".format(nspec, len(qsos.z_qsos)))
        this_flux        = qsos.find_this_flux(nspec)
        this_wavelengths = qsos.find_this_wavelengths(nspec)

        lya1pz = this_wavelengths / lya_wavelength 

        # find the skyline pixels
        inds = np.abs(this_flux[:-1] - this_flux[1:]) > min_flux_thresh

        neg_inds = this_flux[:-1][inds] < - min_flux_thresh

        if not np.any(inds) or not np.any(neg_inds):
            continue

        # find the corresponding zabs
        this_zabs = lya1pz[:-1][inds][neg_inds] - 1

        # find the map_z_dlas
        this_z_dlas = qsos.all_z_dlas[nspec, :]
        this_z_dlas = this_z_dlas[~np.isnan(this_z_dlas)]

        if len(this_z_dlas) < 1:
            continue

        for this_z_dla in this_z_dlas:
            delta_zabs = this_zabs - this_z_dla

            if np.any(np.abs(delta_zabs) < min_z_seperation ):
                skyline_dlas.append(nspec)
                continue