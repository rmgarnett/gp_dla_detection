'''
sbatch_union.py : re-combine all pieces of .mat files generated from sbatch
splitted files.

# Note: hdf5storage is used to generate MATLAB compatible file
# Note: all these functions required large memory; for save2mat73_zpatch
#   the peak memory usage would rise to ~150G.
'''
import h5py
import numpy as np
from .qso_loader import QSOLoader

def mat_combine(processed_files, out_filename, chunk_size, maxshape_size):
    '''
    combine pieces of `.mat` files into a `out_filename.mat` file.
    the size of each piece needs to be specified as chunk_size.

    Parameters:
    ----
    `processed_files` (list(str)) : a list of filenames of splitted .mat files you want to combine
    `out_filename`    (str)       : the output filename
    `chunk_size`      (int)       : the size of a piece of splitted array you want to combine. 
        this `chunk_size` is actually just used for searching the keys you want to combine 
        in the first file in processed_files. So just make sure you have the correct `chunk_size` 
        for the first file.
    `maxshape_size`   (int)       : the final size of the reunited `.mat` file.
    '''
    # find keys to append (the key with values equal to chunksize)
    all_filehandles = [h5py.File(processed_file, 'r') for processed_file in processed_files]
    filehandle = all_filehandles[0]
    keys_to_append = [key for key in filehandle.keys() if filehandle[key].shape[-1] == chunk_size]

    # create a new h5 file to append
    out = h5py.File(out_filename, 'w')
    assert np.sum( [f[keys_to_append[0]].shape[-1] for f in all_filehandles] ) == maxshape_size

    for i,f in enumerate(all_filehandles):
        if i == 0: # copy entire dataset in the first occurrence
            for key in f.keys():
                out.create_dataset(
                    key, shape=f[key].shape, maxshape=f[key].shape[:-1] + (None, ), dtype=f[key].dtype)
                out[key][()] = f[key][()]

        else:      # append the remain pieces of dataset into the out h5 file
            for key in keys_to_append:
                out_array    = out[key][()] # this will copy an array from h5 to the variable

                # change the maxshape of 
                new_size = out_array.shape[-1] + f[key].shape[-1]
                axis     = len( out_array.shape ) - 1

                out[key].resize(new_size, axis=axis)

                out[key][()] = np.concatenate(
                    (out_array, f[key][()]), axis=axis)

                del out_array


    assert 0.95 < out['model_posteriors'][:, -1].sum() <= 1.01
    assert out['p_dlas'].shape[-1] == maxshape_size

    out.close()

def save2mat73(filename, out_filename, small_file=False, dla_nhi_cut=False, sample_file="dla_samples_a03.mat"):
    '''
    convert HDF5 saved by h5py to MATLAB compatible format

    Parameters:
    ----
    small_file  (bool) : True if you don't want to save sample_log_likelihoods
    '''
    import hdf5storage

    f = h5py.File(filename)

    processed_file = {}

    for key in f.keys():
        if small_file:
            if "sample_log_likelihoods" in key or "base_sample_inds" in key:
                continue
        processed_file[u'{}'.format(key)] = np.transpose( f[key][()] )

    hdf5storage.write(processed_file, '.', out_filename, matlab_compatible=True)

def save2mat73_zpatch(filename, catalog_file, out_filename, small_file=False, snr_file=None, out_snr_file=None, occams=True):
    '''
    updating the arrays in the file with zwarning flags; 
    assumed you already have zwarning flags in `catalog.mat`

    Note: if you are using occams arg, we are assuming 
        model_posteriors = [ p_no_dlas, p_lls, p_dla_1, p_dla_2, p_dla_3, ... ]
    '''
    import hdf5storage

    f = h5py.File(filename, 'r')
    catalog = h5py.File(catalog_file, 'r')

    filter_flags = catalog['filter_flags'][0, :]

    test_ind = f['test_ind'][0, :].astype(np.bool)
    size = test_ind.sum()

    assert len(test_ind) == len(filter_flags)

    zwarn_ind = filter_flags[test_ind]
    zwarn_ind = (zwarn_ind == 0)

    # update the zwarning flag
    test_ind_zwarn = test_ind * (filter_flags == 0)

    processed_file = {}

    for key in f.keys():
        # if you want small file then you don't want every samples per spectrum
        if small_file:
            if "sample_log_likelihoods" in key or "base_sample_inds" in key:
                continue
        # modify arrays based on the filter_flags
        if f[key].shape[-1] == size:
            array = np.transpose( f[key][()] )
            array = array[zwarn_ind]
            if occams:
                # post-processing model_posteriors with an additional occam's razor
                if key == "model_posteriors":
                    array = QSOLoader._occams_model_posteriors(array, occams_razor=10000)

                    # update p_dlas
                    p_no_dlas = array[:, 0]
                    p_lls     = array[:, 1]
                    p_dlas    = 1 - p_no_dlas - p_lls

                    # add to the processed_file
                    processed_file[u'p_dlas']    = p_dlas[:, None]
                    processed_file[u'p_lls']     = p_lls[:, None]
                    processed_file[u'p_no_dlas'] = p_no_dlas[:, None]

                    del p_dlas, p_lls, p_no_dlas
                elif key in ("p_dlas", "p_lls", "p_no_dlas"):
                    continue # this is ugly but works for now

                if key in ('log_likelihoods_dla', 'log_likelihoods_lls', 'log_posteriors_dla', 'log_posteriors_lls',
                        'sample_log_likelihoods_dla', 'sample_log_likelihoods_lls'):
                    array = array - np.log(10000)
 
            processed_file[u'{}'.format(key)] = array
            del array # delete this array from memory

        elif key == "test_ind":
            processed_file[u'{}'.format(key)] = test_ind_zwarn[:, None]

        # not per spec property or test_ind, just append them to the processed_file
        else:
            processed_file[u'{}'.format(key)] = np.transpose( f[key][()] )

    if occams:
        # test if the occams rescaling is correct
        model_posteriors = np.append( 
            [processed_file['log_posteriors_no_dla'][0, 0], processed_file['log_posteriors_lls'][0, 0]], 
            processed_file['log_posteriors_dla'][0, :])
        max_posteriors   = np.max(model_posteriors)

        model_posteriors = np.exp(model_posteriors - max_posteriors)

        model_posteriors = model_posteriors / np.sum(model_posteriors)
        errors = np.sum( np.abs(model_posteriors - processed_file['model_posteriors'][0, :]) )
        assert errors < 0.01

    # write into matlab compatible file
    hdf5storage.write(processed_file, '.', out_filename, matlab_compatible=True)

    if snr_file:
        snr = {}
        f_snr = h5py.File(snr_file, 'r')

        array = f_snr['snrs'][0, :]
        array = array[zwarn_ind]

        snr[u'{}'.format('snrs')] = array[:, None]
        hdf5storage.write(snr, '.', out_snr_file, matlab_compatible=True)
