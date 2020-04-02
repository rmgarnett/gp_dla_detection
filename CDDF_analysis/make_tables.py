"""Load tabulated results from text files and output latex tables"""
import numpy as np
import glob
import re
import os

def format_latex_num(number, prec=3, trans = -3):
    """Return a strong formatting a number as, eg 3.1 x 10^4"""
    if number == 0.:
        return "$0$"
    exponent = int(np.floor(np.log10(number)))
    if 1 >= exponent > trans:
        forstr = "$ {0:."+str(prec)+"f} $"
        return forstr.format(number)
    else:
        return str("$ {0:.2f} \\times 10^{{ {1:d} }}$").format(number/10**exponent,exponent)

def format_latex_two_num(number,number2, prec=3,trans=-3):
    """Return a strong formatting two numbers as, eg 3.1 -- 2.5 x 10^4"""
    if number == 0.:
        return "$0 - "+format_latex_num(number2)[1:]
    exponent = int(np.min(np.floor(np.log10([number,number2]))))
    if 1 >= exponent > trans:
        forstr = "$ {0:."+str(prec)+"f} - {1:."+str(prec)+"f} $"
        return forstr.format(number, number2)
    else:
        return str("$ [{0:.2f}  - {1:.2f} ]\\times 10^{{ {2:d} }}$").format(number/10**exponent,number2/10**exponent,exponent)

def load_table(txtname, colheaders = None, caption="", omega=False):
    """Load a table and output Latex"""
    table = np.loadtxt(txtname).T
    prec = 4
    if omega:
        table[:,2:]*=1000
        prec = 3
    (nrow,ncol) = np.shape(table)
    assert ncol == len(colheaders)+4
    table_string = "\\begin{table*} \n \\centering \n"
    table_string += "\\begin{tabular}{"+'c'*ncol+"}\n"
    table_string += "\\hline\n"
    #Write headers
    table_string += colheaders[0]
    for ch in colheaders[1:]:
        table_string +=" & " + ch
    table_string += " & "+"$68$\% limits"
    table_string += " & "+"$95$\% limits"
    table_string +=" \\\\ \n \hline \n"
    xerr = (table[1,0] - table[0,0])/2.
    assert xerr < 1.
    for row in table:
        #Print first two
        table_string+= format_latex_two_num(row[0]-xerr,row[0] + xerr,prec=2)
        table_string+= " & "+format_latex_num(row[1],prec=prec)
        #Now print errors
        table_string+= " & "+format_latex_two_num(row[2],row[3],prec=prec)
        table_string+= " & "+format_latex_two_num(row[4],row[5],prec=prec)
        table_string += "  \\\\ \n"
    table_string += "\\hline \n  \\end{tabular}\n "
    table_string += "\\caption{"+caption+"}\n"
    table_string += "\\label{tab:"+txtname+"}\n \\end{table*}\n"
    return table_string

def load_cddf_table(txtname, caption=""):
    """Load a table and output Latex"""
    table = np.loadtxt(txtname).T
    (nrow,ncol) = np.shape(table)
    table_string = "\\begin{table*} \n \\centering \n"
    table_string += "\\begin{tabular}{"+'c'*ncol+"}\n"
    table_string += "\\hline\n"
    #Write headers
    scalefact = 1e-21
    scalestr = " $( 10^{{ {0:d} }} )$".format(int(np.log10(scalefact)))
    table_string += "$\log_{10} \mathrm{N}_\mathrm{HI}$ & $f(N_\mathrm{HI})$ " + scalestr
    table_string += " & "+"$68$\% limits" + scalestr
    table_string += " & "+"$95$\% limits" + scalestr
    table_string +=" \\\\ \n \hline \n"
    xerr = (table[1,0] - table[0,0])/2.
    for row in table:
        if row[1] == row[3] == row[5] == 0.:
            break
        #Print first two
        table_string+= format_latex_two_num(row[0]-xerr,row[0] + xerr,prec=1)
        table_string+= " & "+format_latex_num(row[1]/scalefact,trans=-2)
        #Now print errors
        table_string+= " & "+format_latex_two_num(row[2]/scalefact,row[3]/scalefact,trans=-2)
        table_string+= " & "+format_latex_two_num(row[4]/scalefact,row[5]/scalefact,trans=-2)
        table_string += "  \\\\ \n"
    table_string += "\\hline \n  \\end{tabular}\n "
    table_string += "\\caption{"+caption+"}\n"
    table_string += "\\label{tab:"+txtname+"}\n \\end{table*}\n"
    return table_string

# load_table("DR12/dndx_all.txt",colheaders = ("z", "dN/dX", "lower 1 sigma", "upper 1 sigma","lower 2 sigma", "upper 2 sigma"),caption="Table of dN/dX values")

def print_all_tables():
    """Print latex for all tables"""
    print(load_table("DR12/dndx_all.txt",colheaders = ("$z$", "dN/dX"),caption="Table of dN/dX values"))
    print(load_table("DR12/omega_dla_all.txt",colheaders = ("$z$", "$\Omega_\mathrm{DLA} (10^{-3}) $"),caption="$\Omega_\mathrm{DLA}$ values",omega=True))
    fNstr = "$f(N_\mathrm{HI})$ "
    ctxts = glob.glob("DR12/cddf_*.txt")
    for ctxt in ctxts:
        print(load_cddf_table(ctxt,caption="CDDF"))

def print_all_multi_dlas_tables(path):
    '''
    print all the tables in the multi-DLA paper:
    - dN/dX table
    - OmegaDLA table
    - CDDF tables : including 1) all; 2) table for different redshifts
    '''
    print(load_table(os.path.join(path, "dndx_all.txt"), colheaders=('$z$', 'dN/dX'), caption="Table of dN/dX values"))
    print(load_table(os.path.join(
        path, "omega_dla_all.txt"), colheaders=('$z$', '$\Omega_\mathrm{DLA} (10^{-3}) $'), caption="$\Omega_\mathrm{DLA}$ values", omega=True))

    fNstr = "$f(N_\mathrm{HI})$ "
    ctxts = glob.glob(os.path.join(path, 'cddf_*.txt'))

    for ctxt in ctxts:
        print(load_cddf_table(ctxt, caption="CDDF"))
