"""
Make some plots for the cosmology-from-home presentation.
"""
import matplotlib
from matplotlib import pyplot as plt

from .qso_loader import QSOLoader

save_figure = lambda filename : plt.savefig("{}.svg".format(filename), format="svg")

def make_this_mu_plots(qsos: QSOLoader, nspec: int = 140845):
    """
    make custom plots
    """
    # null model
    log_p = qsos.processed_file['log_posteriors_no_dla'][0, nspec]
    qsos.plot_this_mu(
        nspec,
        suppressed=False,
        num_voigt_lines=0,
        ls="--",
        label="null model before effective optical depth",
    )
    qsos.plot_this_mu(
        nspec, suppressed=True, num_voigt_lines=0, new_fig=False,
        label="null: log posterior = {:.3g}".format(log_p)
    )
    plt.ylim(-1, 5)
    save_figure("slide_this_mu_null")
    plt.show()

    # 1 DLA model
    log_p = qsos.processed_file['log_posteriors_dla'][0, nspec]
    qsos.plot_this_mu(
        nspec,
        suppressed=False,
        num_voigt_lines=0,
        ls="--",
        label="null model before effective optical depth",
    )
    qsos.plot_this_mu(
        nspec,
        suppressed=True,
        num_voigt_lines=3,
        new_fig=False,
        label=r"$\mathcal{M}$ DLA(1): log posterior = " + "{:.3g}".format(log_p),
        num_dlas=1,
    )
    plt.ylim(-1, 5)
    save_figure("slide_this_mu_DLA1")
    plt.show()


    # 2 DLA model
    log_p = qsos.processed_file['log_posteriors_dla'][1, nspec]
    qsos.plot_this_mu(
        nspec,
        suppressed=False,
        num_voigt_lines=0,
        ls="--",
        label="null model before effective optical depth",
    )
    qsos.plot_this_mu(
        nspec,
        suppressed=True,
        num_voigt_lines=3,
        new_fig=False,
        label=r"$\mathcal{M}$ DLA(2): log posterior = " + "{:.3g}".format(log_p),
        num_dlas=2,
    )
    plt.ylim(-1, 5)
    save_figure("slide_this_mu_DLA2")
    plt.show()

    # 3 DLA model
    log_p = qsos.processed_file['log_posteriors_dla'][2, nspec]
    qsos.plot_this_mu(
        nspec,
        suppressed=False,
        num_voigt_lines=0,
        ls="--",
        label="null model before effective optical depth",
    )
    qsos.plot_this_mu(
        nspec,
        suppressed=True,
        num_voigt_lines=3,
        new_fig=False,
        label=r"$\mathcal{M}$ DLA(3): log posterior = " + "{:.3g}".format(log_p),
        num_dlas=3,
    )
    plt.ylim(-1, 5)
    save_figure("slide_this_mu_DLA3")
    plt.show()
