# -*- coding: utf-8 -*-
'''
Module to load `preloaded_qsos_{release}.mat` and `learned_qso_{training_set}.mat`,
    also load part of the `processed_qsos_{test_set}.mat` for plotting examples and 
    comparing with Parks (2018) catalogue.

Plotting for papers:
- The ROC curve : could be plotted by comparing with the concordance catalogue (visual inspect + 
    Noterdaeme + Fisher method)
- The confusion matrix
- Some examples for the model modifications:
  - learned M
  - mean flux changes
  - an example of mean-flux fitting at high redshifts
- Some sample spectra with multi-DLAs detections, w/ p(y | theta, Model) -- probably need to write
    into another .mat file to stores all predicted dla Voigt profiles
'''
import os
from collections import namedtuple, Counter
import h5py
import numpy as np 
from scipy import integrate
from scipy.special import logsumexp
from matplotlib import pyplot as plt
from .set_parameters import *
from .calc_cddf import HubbleByH0, path_length_int
from .voigt import Voigt_absorption

tau_lyseries = lambda tau, oscillator_strength, transition_wavelength : (tau * 
    oscillator_strength / lya_oscillator_strength * transition_wavelength / lya_wavelength)

class GPLoader(object):
    '''
    Class to store parameters of our GP model
    '''
    tau_0_kim = 0.0023
    beta_kim  =   3.65

    def __init__(self, rest_wavelengths, mu, M, log_tau_0, log_beta, log_c_0, log_omega):
        self.rest_wavelengths = rest_wavelengths
        self.mu = mu
        self.M  = M
        self.log_tau_0 = log_tau_0
        self.log_beta  = log_beta
        self.log_c_0   = log_c_0
        self.log_omega = log_omega

        self.C = self.build_correlation_matrix(M)

    @staticmethod
    def build_correlation_matrix(M):
        '''
        Covert covariance matrix to correlation matrix

        Parameters:
        ----
        M (N pixels, k) : K = M M' , K is a covariance matrix, M is its matrix decomposition   

        Return:
        ----
        C : correlations matrix with its diag C = I
        '''
        # build covariance matrix
        K = np.matmul( M, M.T )

        # query diag elements
        d = np.sqrt(np.diag( K ))[:, np.newaxis]
        
        M_div_d = M / d

        C = np.matmul( M_div_d, M_div_d.T)

        return C


class QSOLoader(object):
    '''
    Class to read preloaded QSO spectroscopic data, D = {λ, y},
    '''
    def __init__(self, preloaded_file="preloaded_qsos.mat", catalogue_file="catalog.mat", 
        learned_file="learned_qso_model_dr9q_minus_concordance.mat", processed_file="processed_qsos_dr12q.mat",
        dla_concordance="dla_catalog", los_concordance="los_catalog", snrs_file="snrs_qsos_dr12q.mat",
        sub_dla=True, sample_file="dla_samples.mat", occams_razor=10000):
        self.preloaded_file = h5py.File(preloaded_file, 'r')
        self.catalogue_file = h5py.File(catalogue_file, 'r')
        self.learned_file   = h5py.File(learned_file,   'r')
        self.processed_file = h5py.File(processed_file, 'r')
        self.snrs_file      = h5py.File(snrs_file,      'r')

        self.sub_dla = sub_dla
        self.occams_razor = occams_razor

        # test_set prior inds : organise arrays into the same order using selected test_inds
        self.test_ind = self.processed_file['test_ind'][0, :].astype(np.bool) #size: (num_qsos, )
        self.test_real_index = np.nonzero( self.test_ind )[0]        

        # load processed data
        self.model_posteriors = self.processed_file['model_posteriors'][()].T

        self.p_dlas           = self.processed_file['p_dlas'][0, :]
        self.p_no_dlas        = self.processed_file['p_no_dlas'][0, :]
        if self.sub_dla:
            self.p_no_dlas += self.model_posteriors[:, 1]
        self.log_priors_dla   = self.processed_file['log_priors_dla'][0, :]
        self.min_z_dlas       = self.processed_file['min_z_dlas'][0, :]
        self.max_z_dlas       = self.processed_file['max_z_dlas'][0, :]
        if 'MAP_log_nhis' in self.processed_file.keys():
            self.map_log_nhis = self.processed_file['MAP_log_nhis'][()].T
            self.map_z_dlas   = self.processed_file['MAP_z_dlas'][()].T

        # load snrs data
        self.snrs             = self.snrs_file['snrs'][0, :]

        # store thing_ids based on test_set prior inds
        self.thing_ids = self.catalogue_file['thing_ids'][0, :].astype(np.int)
        self.thing_ids = self.thing_ids[self.test_ind]

        # plates, mjds, fiber_ids
        self.plates    = self.catalogue_file['plates'][0, :].astype(np.int)
        self.mjds      = self.catalogue_file['mjds'][0, :].astype(np.int)
        self.fiber_ids = self.catalogue_file['fiber_ids'][0, :].astype(np.int)

        self.plates    = self.plates[self.test_ind]
        self.mjds      = self.mjds[self.test_ind]
        self.fiber_ids = self.fiber_ids[self.test_ind]

        # store small arrays
        self.z_qsos     = self.catalogue_file['z_qsos'][0, :]
        self.snrs_cat   = self.catalogue_file['snrs'][0, :]

        self.z_qsos = self.z_qsos[self.test_ind]
        self.snrs_cat   = self.snrs_cat[self.test_ind]

        # [Occams Razor] Update model posteriors with an additional occam's razor
        # updating: 1) model_posteriors, p_dlas, p_no_dlas
        self.model_posteriors = self._occams_model_posteriors(self.model_posteriors, self.occams_razor)
        self.p_dlas    = self.model_posteriors[:, 1+self.sub_dla:].sum(axis=1)
        self.p_no_dlas = self.model_posteriors[:, :1+self.sub_dla].sum(axis=1)

        # build a MAP number of DLAs array
        # construct a reference array of model_posteriors in Roman's catalogue for computing ROC curve
        multi_p_dlas    = self.model_posteriors # shape : (num_qsos, 2 + num_dlas)

        dla_map_model_index = np.argmax( multi_p_dlas, axis=1 )
        multi_p_dlas = multi_p_dlas[ np.arange(multi_p_dlas.shape[0]), dla_map_model_index ]

        # remove all NaN slices from our sample
        nan_inds = np.isnan( multi_p_dlas )

        self.test_ind[self.test_ind == True] = ~nan_inds

        multi_p_dlas          = multi_p_dlas[~nan_inds]
        dla_map_model_index   = dla_map_model_index[~nan_inds]
        self.test_real_index  = self.test_real_index[~nan_inds]
        self.model_posteriors = self.model_posteriors[~nan_inds, :]
        self.p_dlas           = self.p_dlas[~nan_inds]
        self.p_no_dlas        = self.p_no_dlas[~nan_inds]
        self.min_z_dlas       = self.min_z_dlas[~nan_inds]
        self.max_z_dlas       = self.max_z_dlas[~nan_inds]
        if 'MAP_log_nhis' in self.processed_file.keys():
            self.map_log_nhis = self.map_log_nhis[~nan_inds, :, :]
            self.map_z_dlas   = self.map_z_dlas[~nan_inds, :, :]
        self.thing_ids        = self.thing_ids[~nan_inds]
        self.plates           = self.plates[~nan_inds]
        self.mjds             = self.mjds[~nan_inds]
        self.fiber_ids        = self.fiber_ids[~nan_inds]
        self.z_qsos           = self.z_qsos[~nan_inds]
        self.snrs_cat         = self.snrs_cat[~nan_inds]
        self.snrs             = self.snrs[~nan_inds]
        self.log_priors_dla   = self.log_priors_dla[~nan_inds]

        self.nan_inds = nan_inds
        assert np.any( np.isnan( multi_p_dlas )) == False

        # get the number of DLAs with the highest val in model_posteriors
        dla_map_num_dla = dla_map_model_index
        if self.sub_dla:
            # (no dla, sub dla, DLA(1), DLA(2), ..., DLA(k))
            dla_map_num_dla = dla_map_num_dla - self.sub_dla
            dla_map_num_dla[dla_map_num_dla < 0] = 0

        self.multi_p_dlas        = multi_p_dlas
        self.dla_map_model_index = dla_map_model_index
        self.dla_map_num_dla     = dla_map_num_dla

        # construct an array based on model_posteriors, the array should be
        # [ {P(Mdla(m) | D)}j=1^m, P(Mdla(m+1) | D), ..., P(Mdla(k) | D) ]
        # This should be the reference p_dlas for multi-DLAs if we divide each los into num_dla pieces.
        model_posteriors_dla = np.copy( self.model_posteriors[:, 1 + self.sub_dla:].T ) # shape : (num_dlas, num_qsos)
        num_models = model_posteriors_dla.shape[0]
        num_qsos   = model_posteriors_dla.shape[1]

        # build a mask array for assign P(Mdla(m) | D) until `m`th index
        indices = (np.arange(1, num_models + 1, dtype=np.int8)[:, None] * np.ones( num_qsos, dtype=np.int8 ))
        multi_dla_map_num_dla   = (self.dla_map_num_dla * np.ones( (num_models, num_qsos), dtype=np.int8 )) # shape : (num_dlas, num_qsos) )
        multi_highest_posterior = (self.multi_p_dlas * np.ones( (num_models, num_qsos), dtype=np.int8 ))
        multi_p_no_dlas         = (self.p_no_dlas    * np.ones( (num_models, num_qsos), dtype=np.int8 ))
        mask_inds = indices <= multi_dla_map_num_dla
        
        model_posteriors_dla[mask_inds] = multi_highest_posterior[mask_inds]
        self.model_posteriors_dla = model_posteriors_dla
        self.multi_p_no_dlas      = multi_p_no_dlas

        # store learned GP models
        self.GP = GPLoader(
            self.learned_file['rest_wavelengths'][:, 0],
            self.learned_file['mu'][:, 0],
            self.learned_file['M'][()].T,
            self.learned_file['log_tau_0'][0, 0],
            self.learned_file['log_beta'][0, 0],
            self.learned_file['log_c_0'][0, 0],
            self.learned_file['log_omega'][:, 0]
        )

        # load dla_catalog
        self.load_dla_concordance(dla_concordance, los_concordance)

        # get MAP vals
        # `MAP_log_nhis` only exists after my modification of roman's code
        # so I should write another method to load roman's catalogue's MAPs
        if 'MAP_log_nhis' in self.processed_file.keys():
            self.prepare_map_vals()
        # This works but it is super slow
        else:
            self.sample_file = sample_file
        #     self.prepare_roman_map_vals(sample_file=sample_file)
        

        # make sure everything sums to unity
        assert np.all( 
            (self.model_posteriors.sum(axis=1) < 1.2) * 
            (self.model_posteriors.sum(axis=1) > 0.8) )
        
    @staticmethod
    def _occams_model_posteriors(model_posteriors, occams_razor=10000):
        '''
        re-calculate the model posteriors based on an additional occams_razor penaly
        
        P(DLA | D) = P(DLA | D) / occams_razor 
                    / ( P(noDLA | D) + P(DLA | D) / occams_razor + P(subDLAs | D) / occams_razor  )
        
        Parameters:
        ----
        model_posteriors (np.ndarray) : shape (num_qsos, sub_dla + noDLA + k DLAs)
        
        '''
        # all subDLAs + DLAs needed to be normalised
        model_posteriors[:, 1:] = model_posteriors[:, 1:] / occams_razor

        # calculate normalisation factor
        normalisation = np.sum( model_posteriors, axis=1)[:, None] * np.ones(model_posteriors.shape[1])[None, :]

        model_posteriors = model_posteriors / normalisation

        assert np.all( (0.8 < np.sum(model_posteriors, axis=1)) & (np.sum(model_posteriors, axis=1) < 1.2))

        return model_posteriors      


    def reevaluate_model_posteriors(self):
        '''
        With some reasons, model_posteriors will happen to be not summed to unity.
        One possibility is the issue related to floating points.
        We re_evalute the model_posteriors here using log_posteriors.
        '''
        if np.where( self.model_posteriors.sum(axis=1) > 1.2 )[0].shape[0] != 0:
            log_posteriors_no_dla = self.processed_file['log_posteriors_no_dla'][()].T
            log_posteriors_lls    = self.processed_file['log_posteriors_lls'][()].T
            log_posteriors_dla    = self.processed_file['log_posteriors_dla'][()].T

            log_posteriors = np.concatenate(
                (log_posteriors_no_dla, log_posteriors_lls, log_posteriors_dla),
                axis=1)

            max_log_posteriors = log_posteriors.max(axis=1)

            model_posteriors = np.exp(
                log_posteriors - max_log_posteriors[:, None]
            )
            model_posteriors = model_posteriors / model_posteriors.sum(axis=1)[:, None]

            self.model_posteriors = model_posteriors
            self.log_posteriors = log_posteriors

    def prepare_map_vals(self):
        '''
        create two MAP arrays to store MAP values of log nhis and z_dlas
        '''
        self.all_log_nhis = np.empty( self.map_log_nhis.shape[:2] )
        self.all_z_dlas   = np.empty( self.map_z_dlas.shape[:2]   )

        self.all_log_nhis[:] = np.nan
        self.all_z_dlas[:]   = np.nan

        for i, nth in enumerate(self.dla_map_num_dla):
            # skip Model(no dla)
            if nth == 0:
                continue
            
            self.all_log_nhis[i, :nth] = self.map_log_nhis[i, nth - 1, :nth]
            self.all_z_dlas[i, :nth]   = self.map_z_dlas[i, nth - 1, :nth]
            
    def prepare_roman_map_vals(self, sample_file="dla_samples.mat", use_memory=False, split=10):
        '''
        create two MAP arrays to store MAP values of log nhi and z_dla in roman's 
        single DLA per spec catalogue.
        '''
        if 'sample_log_likelihoods_dla' not in self.processed_file.keys():
            raise Exception(KeyError, 'Should use the processed file with sample likelihoods.')

        # make sure we only have pDLA and pnoDLA
        assert self.model_posteriors.shape[1] == 2 

        # initialize the arrays
        self.all_log_nhis = np.empty( self.thing_ids.shape )
        self.all_z_dlas   = np.empty( self.thing_ids.shape )

        self.all_log_nhis[:] = np.nan
        self.all_z_dlas[:]   = np.nan

        # find MAP values manually
        sample_filehandle = h5py.File(sample_file, 'r')
        log_nhi_samples   = sample_filehandle['log_nhi_samples'][:, 0]
        offset_samples    = sample_filehandle['offset_samples'][:, 0]

        if use_memory:
            chunk  = len(self.thing_ids) // split
            length = len(self.thing_ids)

            for init_nspec,end_nspec in zip(
                    range(0, length, chunk),
                    range(chunk, length + chunk, chunk)):
                # load everything into memory first and then do the operations
                # handling the NaN vals
                real_index_slice = np.nonzero(~self.nan_inds)[0][init_nspec:end_nspec]
                likelihoods = self.processed_file['sample_log_likelihoods_dla'][:, real_index_slice]
                likelihoods = likelihoods.T

                assert likelihoods.shape[1] == log_nhi_samples.shape[0]

                map_index = likelihoods.argmax(axis=1)

                this_offset_samples = offset_samples[map_index]

                sample_z_dlas = self.min_z_dlas[init_nspec:end_nspec] + (
                    self.max_z_dlas[init_nspec:end_nspec] - 
                    self.min_z_dlas[init_nspec:end_nspec]) * this_offset_samples

                assert sample_z_dlas.shape[0] == len(map_index)

                self.all_log_nhis[init_nspec:end_nspec] = log_nhi_samples[map_index]
                self.all_z_dlas[init_nspec:end_nspec]   = sample_z_dlas

                del likelihoods

                print("preparing MAP vals : {} / {}".format( end_nspec, length ))

        else:
            # find the MAP values iteratively to avoid memory issue
            for nspec in range(self.thing_ids.shape[0]):
                # prepare the sample values
                sample_z_dlas = self.min_z_dlas[nspec] + (
                    self.max_z_dlas[nspec] - self.min_z_dlas[nspec]) * offset_samples

                # load into memory first and then do the operations
                i = np.nonzero(~self.nan_inds)[0][nspec]
                likelihoods = self.processed_file['sample_log_likelihoods_dla'][:, i]

                map_index = likelihoods.argmax()

                # query the MAP values and store
                self.all_log_nhis[nspec] = log_nhi_samples[map_index]
                self.all_z_dlas[nspec]   = sample_z_dlas[map_index]

    def prepare_roam_map_vals_per_spec(self, nspec, sample_file="dla_samples.mat"):
        '''
        query the MAP values for Roman's single DLA catalogue
        '''
        # make sure we only have pDLA and pnoDLA
        assert self.model_posteriors.shape[1] == 2

        if "all_log_nhis" not in dir(self):
            # initialize the arrays
            self.all_log_nhis = np.empty( self.thing_ids.shape )
            self.all_z_dlas   = np.empty( self.thing_ids.shape )

            self.all_log_nhis[:] = np.nan
            self.all_z_dlas[:]   = np.nan

        # find MAP values manually
        sample_filehandle = h5py.File(sample_file, 'r')
        log_nhi_samples   = sample_filehandle['log_nhi_samples'][:, 0]
        offset_samples    = sample_filehandle['offset_samples'][:, 0]

        # prepare the sample values
        sample_z_dlas = self.min_z_dlas[nspec] + (
            self.max_z_dlas[nspec] - self.min_z_dlas[nspec]) * offset_samples

        # load into memory first and then do the operations
        i = np.nonzero(~self.nan_inds)[0][nspec]
        likelihoods = self.processed_file['sample_log_likelihoods_dla'][:, i]

        map_index = likelihoods.argmax()

        # query the MAP values and store
        self.all_log_nhis[nspec] = log_nhi_samples[map_index]
        self.all_z_dlas[nspec]   = sample_z_dlas[map_index]


    def load_dla_concordance(self, dla_concordance, los_concordance, lnhi_min=20, release='dr9'):
        '''
        load dla_concordance .txt file : (thing_ids, z_dlas, log_nhis)
        
        Also, match the existed thing_ids in the test data (processed data)

        Parameters:
        ----
        dla_concordance (str) : path to the concordance DLA catalogue
        los_concordance (str) : path to the concordance LOS catalogue
        lnhi_min (float) : minimum value logNHI for DLAs
        release (str) : default 'dr9'
        '''
        dla_catalog = np.loadtxt(dla_concordance)
        los_catalog = np.loadtxt(los_concordance)

        thing_ids = dla_catalog[:, 0].astype(np.int)
        z_dlas    = dla_catalog[:, 1]
        log_nhis  = dla_catalog[:, 2]

        thing_ids_los = los_catalog.astype(np.int)

        # apply the lower cut for logNHI
        inds = log_nhis > lnhi_min

        thing_ids = thing_ids[inds]
        z_dlas    = z_dlas[inds]
        log_nhis  = log_nhis[inds]

        assert all(z_dlas < 7) # make sure getting the correct column

        # assumption here is concordance only has 1-DLA
        real_index = np.where( np.in1d(self.thing_ids, thing_ids) )[0]
        real_index_los = np.where( np.in1d(self.thing_ids, thing_ids_los) )[0]

        # re-select (z_dla, log_nhi) values from the intersection bt Garnett
        inds = np.in1d( thing_ids, self.thing_ids )
        z_dlas   = z_dlas[inds]
        log_nhis = log_nhis[inds]

        assert z_dlas.shape[0] == real_index.shape[0]

        # assert real_index.shape[0] == thing_ids.shape[0]
        print(
            "[Warning] {} DLAs lost and {} QSOs lost after np.in1d (searching matched thing_ids in the test data).".format(
                thing_ids.shape[0] - real_index.shape[0], 
                thing_ids_los.shape[0] - real_index_los.shape[0]))

        # store data in named tuple under self
        dla_catalog = namedtuple(
            'dla_catalog_concordance', 
            ['real_index', 'real_index_los', 
            'thing_ids', 'thing_ids_los', 
            'z_dlas', 'log_nhis', 'release'])
        self.dla_catalog = dla_catalog(
            real_index=real_index, real_index_los=real_index_los, 
            thing_ids=thing_ids, thing_ids_los=thing_ids_los, 
            z_dlas=z_dlas, log_nhis=log_nhis, release=release)

    def load_dla_parks(self, dla_parks, p_thresh=0.5, release='dr12', multi_dla=True, num_dla=2):
        '''
        load predictions_DR12.json from Parks(2018) catalogue

        Also, matched the existed thing_ids in the test data.
        Note: we have to consider DLAs from the same sightlines as different objects

        Parameters:
        ----
        dla_parks (str) : the filename of Parks (2018) product
        p_thresh (float): the minimum probability to be considered as a DLA in Parks(2018)
        release (str) 
        multi_dla (bool): whether or not we want to construct multi-dla index
        num_dla (int)   : number of dla we want to consider if we are considering multi-dlas

        Note:
        ---
        unique_ids (array) : plates * 10**9 + mjds * 10**4 + fiber_ids,
            this is an unique array constructed for matching between Parks and Roman's catalogues.
            note that make sure it is int64 since we have exceeded 2**32
        '''
        dict_parks = self.prediction_json2dict(dla_parks)

        # construct an array of unique ids for los
        self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
        unique_ids      = self.make_unique_id( dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'])  
        assert unique_ids.dtype is np.dtype('int64')
        assert self.unique_ids.dtype is np.dtype('int64')

        # TODO: make the naming of variables more consistent
        parks_in_garnett_inds = np.in1d( unique_ids, self.unique_ids )
        raw_unique_ids      = unique_ids[parks_in_garnett_inds]
        raw_z_dlas          = dict_parks['z_dlas'][parks_in_garnett_inds]
        raw_log_nhis        = dict_parks['log_nhis'][parks_in_garnett_inds]
        raw_dla_confidences = dict_parks['dla_confidences'][parks_in_garnett_inds]

        real_index_los = np.where( np.in1d(self.unique_ids, unique_ids) )[0]
        
        unique_ids_los = self.unique_ids[real_index_los]
        thing_ids_los  = self.thing_ids[real_index_los]
        assert np.unique(unique_ids_los).shape[0] == unique_ids_los.shape[0] # make sure we don't double count los

        # construct an array of unique ids for dlas
        dla_inds = dict_parks['dla_confidences'] > p_thresh

        real_index_dla = np.where( np.in1d(self.unique_ids, unique_ids[dla_inds]) )[0] # Note that in this step we lose
                                                                              # the info about multi-DLA since 
                                                                              # we are counting based on los

        unique_ids_dla = self.unique_ids[real_index_dla]
        thing_ids_dla  = self.thing_ids[real_index_dla]

        # Construct a list of sub-los index and dla detection based on sub-los.
        # This is a relatively complicate loop and it's hard to understand philosophically.
        # It's better to write an explaination in the paper.
        if multi_dla:
            self.multi_unique_ids = self.make_multi_unique_id(num_dla, self.plates, self.mjds, self.fiber_ids) 
            multi_unique_ids      = self.make_multi_unique_id(
                num_dla, dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'])  # note here some index repeated 
                                                                                             # more than num_dla times

            multi_real_index_los = np.where( np.in1d(self.multi_unique_ids, multi_unique_ids) )[0] # here we have a real_index array
                                                                                                 # exactly repeat num_dla times

            multi_unique_ids_los = self.multi_unique_ids[multi_real_index_los]

            self.multi_thing_ids = self.make_array_multi(num_dla, self.thing_ids)
            multi_thing_ids_los  = self.multi_thing_ids[multi_real_index_los]

            # loop over unique_ids to assign DLA detection to sub-los
            # Note: here we ignore the z_dla of DLAs.
            dla_multi_inds = np.zeros(multi_unique_ids_los.shape, dtype=bool)
            for uid in np.unique(multi_unique_ids_los):
                k_dlas = ( dict_parks['dla_confidences'][unique_ids == uid] > p_thresh ).sum()

                k_dlas_val = np.zeros(num_dla, dtype=bool)
                k_dlas_val[:k_dlas] = True                 # assigning True until DLA(k)

                # assign DLA detections to the unique_ids of sub-los
                dla_multi_inds[ multi_unique_ids_los == uid ] = k_dlas_val
                assert multi_unique_ids_los[ multi_unique_ids_los == uid ].shape[0] == num_dla
                
            multi_real_index_dla = multi_real_index_los[dla_multi_inds]
            multi_unique_ids_dla = multi_unique_ids_los[dla_multi_inds]
            multi_thing_ids_dla  = multi_thing_ids_los[dla_multi_inds]

            # store data in named tuple under self
            dla_catalog = namedtuple(
                'dla_catalog_parks', 
                ['real_index', 'real_index_los', 
                'thing_ids', 'thing_ids_los',
                'unique_ids', 'unique_ids_los',
                'multi_real_index_dla', 'multi_real_index_los',
                'multi_thing_ids_dla', 'multi_thing_ids_los',
                'multi_unique_ids_dla', 'multi_unique_ids_los', 
                'release', 'num_dla', 
                'raw_unique_ids', 'raw_z_dlas', 'raw_log_nhis', 'raw_dla_confidences' ])
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla, real_index_los=real_index_los, 
                thing_ids=thing_ids_dla, thing_ids_los=thing_ids_los, 
                unique_ids=unique_ids_dla, unique_ids_los=unique_ids_los,
                multi_real_index_dla=multi_real_index_dla, multi_real_index_los=multi_real_index_los,
                multi_thing_ids_dla=multi_thing_ids_dla, multi_thing_ids_los=multi_thing_ids_los,
                multi_unique_ids_dla=multi_unique_ids_dla, multi_unique_ids_los=multi_unique_ids_los,
                release=release, num_dla=num_dla,
                raw_unique_ids=raw_unique_ids, raw_z_dlas=raw_z_dlas, 
                raw_log_nhis=raw_log_nhis, raw_dla_confidences=raw_dla_confidences)

        else:
            dla_catalog = namedtuple(
                'dla_catalog_parks', 
                ['real_index', 'real_index_los', 
                'thing_ids', 'thing_ids_los',
                'unique_ids', 'unique_ids_los',
                'release',
                'raw_unique_ids', 'raw_z_dlas', 'raw_log_nhis', 'raw_dla_confidences' ])
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla, real_index_los=real_index_los, 
                thing_ids=thing_ids_dla, thing_ids_los=thing_ids_los, 
                unique_ids=unique_ids_dla, unique_ids_los=unique_ids_los,
                release=release,
                raw_unique_ids=raw_unique_ids, raw_z_dlas=raw_z_dlas, 
                raw_log_nhis=raw_log_nhis, raw_dla_confidences=raw_dla_confidences)

    @staticmethod
    def make_array_multi(num_dla, array):
        '''
        make an array of los to an array with num_dla sub-los. 
        Can be imagined as splitting a single los to num_dla pieces.
        '''
        assert num_dla > 1
        return (np.ones(num_dla)[:, None] * array).ravel()

    @staticmethod
    def make_unique_id(plates, mjds, fiber_ids):
        return plates * 10**9 + mjds * 10**4 + fiber_ids

    def make_multi_unique_id(self, num_dla, plates, mjds, fiber_ids):
        '''
        To count the multi-dlas, it is better to count on the basis of number of 
        dlas instead of number of sightlines. If we count on the number of dlas,
        we will not intrepret a DLA(2) model as a single false positive detection if 
        the truth is DLA(1). Instead, we will have one true positive and one false positive.

        '''
        multi_unique_id = np.ones(num_dla)[:, None] * self.make_unique_id(plates, mjds, fiber_ids)
        assert np.prod(multi_unique_id.shape) == num_dla * plates.shape[0]
        return multi_unique_id.ravel()

    def make_multi_ROC(self, catalog):
        '''
        Make a ROC curve with a given `catalog`, which must contains 
        (num_dla, multi_real_index_dla, multi_real_index_los).

        multi_real_index should be the real index for any arrays passing through 
        self.make_array_multi(num_dla, array)
        '''
        # bool array, this builds an array describing sub-los containing DLAs or not
        # Note: real_index are unique, non-repetitive
        dla_ind = np.in1d( catalog.multi_real_index_los, catalog.multi_real_index_dla ) # shape : flatten( (num_dla, num_los) )

        # treat each single element as one sightline
        multi_model_posteriors_dla = self.model_posteriors_dla.ravel()
        
        p_dlas = multi_model_posteriors_dla[catalog.multi_real_index_los]
        p_no_dlas = self.multi_p_no_dlas.ravel()[catalog.multi_real_index_los]
        
        odds_dla_no_dla = p_dlas / p_no_dlas

        rank_idx = np.argsort( odds_dla_no_dla ) # small odds -> large odds
        assert p_dlas[ rank_idx[0] ] < p_dlas[ rank_idx[-1] ]

        # re-order every arrays based on the rank
        dla_ind         = dla_ind[ rank_idx ]
        odds_dla_no_dla = odds_dla_no_dla[ rank_idx ]

        TPR = []
        FPR = []
        
        for odd in odds_dla_no_dla:
            odd_ind = odds_dla_no_dla >= odd

            true_positives   =  dla_ind &  odd_ind
            false_negatives  =  dla_ind & ~odd_ind
            true_negatives   = ~dla_ind & ~odd_ind
            false_positives  = ~dla_ind &  odd_ind

            TPR.append( np.sum(true_positives) / ( np.sum(true_positives) + np.sum(false_negatives) ) )
            FPR.append( np.sum(false_positives) / (np.sum(false_positives) + np.sum(true_negatives)) )

        return TPR, FPR



    def make_ROC(self, catalog, occams_razor=10000):
        '''
        Make a ROC curve with a given `catalog`, which must contains (real_index, real_index_los)
        '''
        dla_ind = np.in1d( catalog.real_index_los, catalog.real_index ) # boolean array, same size

        # construct an array to rank the posteriors of containing DLAs
        # p_dlas    = self.p_dlas[catalog.real_index_los]
        # p_no_dlas = self.p_no_dlas[catalog.real_index_los]
        # posterior_dlas    = self.model_posteriors[catalog.real_index_los, self.sub_dla + 1:].sum(axis=1)
        # posterior_no_dlas = self.model_posteriors[catalog.real_index_los, 0:1 + self.sub_dla].sum(axis=1)
        #
        # use log_posteriors_dla directly to avoid numerical underflow
        log_posteriors_dla    = self.processed_file['log_posteriors_dla'][()]       - np.log(occams_razor)
        log_posteriors_no_dla = self.processed_file['log_posteriors_no_dla'][0, :]
        if self.sub_dla:
            log_posteriors_lls    = self.processed_file['log_posteriors_lls'][0, :] - np.log(occams_razor)

            # update no DLA posteriors with subDLA posteriors
            log_posteriors_no_dla = logsumexp( [ log_posteriors_no_dla, log_posteriors_lls ], axis=0 )

        log_posteriors_dla = logsumexp(log_posteriors_dla, axis=0)

        # filtering out ~nan_inds
        log_posteriors_dla    = log_posteriors_dla[~self.nan_inds]
        log_posteriors_no_dla = log_posteriors_no_dla[~self.nan_inds]

        # query the corresponding index in the catalog
        log_posteriors_dla    = log_posteriors_dla[catalog.real_index_los]
        log_posteriors_no_dla = log_posteriors_no_dla[catalog.real_index_los]

        odds_dla_no_dla = log_posteriors_dla - log_posteriors_no_dla # log odds

        rank_idx = np.argsort( odds_dla_no_dla ) # small odds -> large odds
        assert log_posteriors_dla[ rank_idx[0] ] < log_posteriors_dla[ rank_idx[-1] ]

        # re-order every arrays based on the rank
        dla_ind = dla_ind[ rank_idx ]
        odds_dla_no_dla = odds_dla_no_dla[rank_idx]

        TPR = []
        FPR = []

        for odd in odds_dla_no_dla:
            odd_ind = odds_dla_no_dla >= odd
            
            true_positives   =  dla_ind &  odd_ind
            false_negatives  =  dla_ind & ~odd_ind
            true_negatives   = ~dla_ind & ~odd_ind
            false_positives  = ~dla_ind &  odd_ind

            TPR.append( np.sum(true_positives) / ( np.sum(true_positives) + np.sum(false_negatives) ) )
            FPR.append( np.sum(false_positives) / (np.sum(false_positives) + np.sum(true_negatives)) )

        return TPR, FPR

    def make_MAP_comparison(self, catalog):
        '''
        make a comparison between map values and concordance values
        
        This is (z_dla_concordance - map_z_dla | concordance ∩ garnett) and
                (log_nhi_concordance - log_nhi | concordance ∩ garnett)
        which means we only consider the difference has overlaps between concordance and ours catalogue
        '''
        # map values array size: (num_qsos, model_DLA(n), num_dlas)
        # use the real_index vals stored in dla_catalog attribute and 
        # loop over the real_index of self.map_values while at the same time
        # get the (MAP_values | DLA(n)) using self.dla_map_model_index
        real_index       = self.dla_catalog.real_index # concordance only

        # get corresponding map vals for concordance only
        map_model_index  = self.dla_map_model_index[real_index]
        
        # make sure having at least one DLA, concordance ∩ garnett
        real_index = real_index[map_model_index > self.sub_dla]
        
        map_z_dlas   = self.map_z_dlas[real_index, 0, 0]   # assume DLA(1) corresponds to the concordance value
        map_log_nhis = self.map_log_nhis[real_index, 0, 0]
        
        Delta_z_dlas   = map_z_dlas   - self.dla_catalog.z_dlas[map_model_index > self.sub_dla]
        Delta_log_nhis = map_log_nhis - self.dla_catalog.log_nhis[map_model_index > self.sub_dla]

        return Delta_z_dlas, Delta_log_nhis

    def make_MAP_parks_comparison(self, catalog, num_dlas=1, dla_confidence=0.98):
        '''
        make a comparison between map values and Park's predictions

        What really computed is:
             (map_z_dla - z_dla_parks    | Parks DLA(1 to n) ∩ garnett DLA(1 to n) ) and
             (map_log_nhi - log_nhi_parks| Parks DLA(1 to n) ∩ garnett DLA(1 to n) ),
        which means
             we only compares spectra containing the same number of DLAs        
        '''
        # Parks contains multiple DLAs, which means we have to identify a way to compare multiple-DLAS
        # within one spectroscopic observations. 
        # We only compare Parks(DLA==n) ∩ Garnett(DLA==n) where n is num_dlas detected within a sightline.
        # The purpose is to test the systematic of mutli-DLA models, though we know Parks have intrinsic bias
        # in estimations of NHI.
        # 
        # We first find indices for Parks(DLA==n) and 
        #     then find indices for Garnett(DLA==n)
        # for multi-DLAs, we find the minimum z_dla value of difference between Parks and Garnett, which is
        #     min( z_dlas[ith qso, Parks(DLA==n)] - z_dlas[ith qso, Garnett(DLA==n)] )
        # while for column density NHI, we use the the DLAs corresponding to minimum z_dlas to compute the MAP difference.
        assert num_dlas > 0

        # filter out low confidence DLAs
        raw_unique_ids  = catalog.raw_unique_ids
        dla_confidences = catalog.raw_dla_confidences
        raw_z_dlas      = catalog.raw_z_dlas
        raw_log_nhis    = catalog.raw_log_nhis

        raw_unique_ids  = raw_unique_ids[dla_confidences > dla_confidence]
        raw_z_dlas      = raw_z_dlas[dla_confidences > dla_confidence]
        raw_log_nhis    = raw_log_nhis[dla_confidences > dla_confidence]

        count_unique_ids = Counter(raw_unique_ids)
        inds  = np.array(list(count_unique_ids.values()), dtype=np.int) == num_dlas
        
        # find IDs | Parks(DLA==n)
        uids_dla_n_parks   =  np.array(list(count_unique_ids.keys()), dtype=np.int)[inds]
        
        # find IDs | Garnett(DLA==n)
        uids_dla_n_garnett = self.unique_ids[self.dla_map_num_dla == num_dlas]
        
        # find IDs | Parks(DLA==n) ∩ Garnett(DLA==n)
        # Note: the intersection for DLA(n==num_dlas) between Parks and Garnett is surprising small.  
        uids_dla_n = uids_dla_n_parks[np.isin(uids_dla_n_parks, uids_dla_n_garnett)]

        inds_dla_n = np.isin( raw_unique_ids, uids_dla_n )

        # for each ID, it has num_dlas elements in the following arrays
        z_dlas_parks     = raw_z_dlas[inds_dla_n]
        log_nhis_parks   = raw_log_nhis[inds_dla_n]
        unique_ids_parks = raw_unique_ids[inds_dla_n]

        # looping over each ID and comparing the MAP values between Parks and Garnett
        Delta_z_dlas   = np.empty((len(uids_dla_n), num_dlas))
        Delta_log_nhis = np.empty((len(uids_dla_n), num_dlas))
        # TODO: to see if there is any way to avoiding using for loop, though it seems to be fine 
        # for multi-DLAs since they are rare events.
        for i,uid in enumerate(uids_dla_n):
            # find Garnett's MAPs(Parks(DLA==n) ∩ Garnett(DLA==n))
            ind = self.unique_ids == uid
            
            this_z_dlas   = self.map_z_dlas[ind][0, num_dlas - 1, :num_dlas]
            this_log_nhis = self.map_log_nhis[ind][0, num_dlas - 1, :num_dlas]

            # find Parks' predictions(Parks(DLA==n) ∩ Garnett(DLA==n))
            ind_parks = unique_ids_parks == uid
            this_z_dlas_parks   = z_dlas_parks[ind_parks]
            this_log_nhis_parks = log_nhis_parks[ind_parks]

            # sort z_dlas to minimize the difference of z_dlas between Parks and Garnett
            argsort_garnett = np.argsort(this_z_dlas) 
            argsort_parks   = np.argsort(this_z_dlas_parks)

            assert np.all(
                    np.abs((this_z_dlas[argsort_garnett] - this_z_dlas_parks[argsort_parks]).sum()) <= 
                    np.abs((this_z_dlas                  - this_z_dlas_parks).sum())
                    )

            Delta_z_dlas[i, :]   = (this_z_dlas[argsort_garnett]   - this_z_dlas_parks[argsort_parks])
            Delta_log_nhis[i, :] = (this_log_nhis[argsort_garnett] - this_log_nhis_parks[argsort_parks])

        return Delta_z_dlas, Delta_log_nhis, z_dlas_parks

    @staticmethod
    def downward_model(posteriors):
        '''
        Remove the final model and re-evaluate the posteriors
        '''
        return posteriors[:-1] / np.sum( posteriors[:-1] )

    def query_least_num_dlas(self, this_model_posteriors, p_thresh):
        '''
        Get the least num_dlas based on the given p_thresh.
        It will loop over from the largest num_dlas in the 
        model. It will stop at the MDLA(k) which has probability
        larger than p_thresh.
        If it couldn't find any P(DLA(k)) > p_thresh, then return
        zero DLAs.
        '''
        tot_num_dlas = len(this_model_posteriors) - 1 - self.sub_dla

        # start with the last posterior
        for i in range(tot_num_dlas):
            posterior = this_model_posteriors[::-1][0]

            if posterior > p_thresh:
                return tot_num_dlas - i

            this_model_posteriors = self.downward_model(this_model_posteriors)

        # if finding no P(DLA(>k)) larger than p_thresh, return no DLA 
        return 0

    @staticmethod
    def query_least_num_dlas_parks(
            uid, raw_unique_ids, raw_dla_confidences, raw_z_dlas, raw_log_nhis, 
            dla_confidence, min_z_dla, min_log_nhi):
        '''
        Get the (num_dlas | Parks, p_dla > dla_confidence, z_dla > min_z_dla, log_nhi > min_log_nhi)
        '''
        # get all the unique_ids corresponding to the given uid
        inds = ( uid == raw_unique_ids )

        # count the number of ids which have p_dla > dla_confidence
        return np.sum(
            (raw_dla_confidences[inds] > dla_confidence) * 
            (raw_z_dlas[inds] > min_z_dla) * 
            (raw_log_nhis[inds] > min_log_nhi)
            )

    def make_multi_confusion(
            self, catalog, dla_confidence=0.98, p_thresh=0.98, hard_cut=False,
            snr=-1, lyb=False, min_log_nhi=20.3):
        '''
        make a confusion matrix for multi-DLA classification

        Parameters:
        ----
        catalog (collections.namedtuple) : assume to be Parks' catalog, but 
            could be extend to other catalogs
        snr : only compare spectra with signal-to-noise > snr
        lyb : only compare DLAs with z_DLA > z_QSO  

        What we want is to compute a confusion matrix for multi-DLAs such as:
        
        Garnett\Parks    no DLA  1DLA    2DLAs   3DLAs
        ----------------------------------------------
        no DLA
        1 DLA
        2 DLAs
        3 DLAs    
        '''
        # Generally, we can loop over the entries in the table one-by-one and 
        # count the numbers on the matrix.
        # It's possible put a hard cut on the model posterior MDLA(k) to drop
        # all the spectra with max(MDLA(k)) < p_thresh, but it will potentially
        # lost lots of spectra we could count in our statistics. 
        # We thus propose to re-evaluate the model_posterior until we get 
        # MDLA(k >= num_dlas) > p_thresh.
        
        # we want unique_ids | Garnett ∩ Parks
        # TODO: generalize this to allow concordance

        # we need to introduce 
        inds = np.isin(self.unique_ids, catalog.raw_unique_ids)
        
        # SNRs cutoff
        inds = inds * ( self.snrs > snr) 

        # Garnett -> Garnett ∩ Parks
        unique_ids = self.unique_ids[inds]
        z_qsos     = self.z_qsos[inds]

        # initialize confusion matrix
        size = self.model_posteriors.shape[1] - self.sub_dla
        confusion_matrix = np.zeros((size, size))

        # matrix to store uid and DLA predictions between Garnett and Parks
        # (uid, # DLAs in Garnett, # DLAs in Parks)
        uid_matrix = np.zeros((len(unique_ids), 3)).astype(np.long)

        for i,uid in enumerate(unique_ids):

            # garnett counts
            inds_g = (uid == self.unique_ids)
            this_model_posteriors = self.model_posteriors[inds_g]
            this_map_z_dlas       = self.map_z_dlas[inds_g][0]
            this_map_log_nhis     = self.map_log_nhis[inds_g][0]
            assert this_model_posteriors.shape[0] == 1

            # z_dla cutoff and lognhi cutoff
            if lyb:
                min_z_dla = (1 + z_qsos[i]) * lyb_wavelength / lya_wavelength - 1
            else:
                min_z_dla = 0

            # put the cutoff of z_dla and lognhi in the query
            n = self.query_least_num_dlas(this_model_posteriors[0], p_thresh)
            if n > 0:
                inds = (this_map_z_dlas[n - 1, :] > min_z_dla) * (this_map_log_nhis[n - 1, :] > min_log_nhi)
                n = np.sum(inds)

            # parks counts
            m = self.query_least_num_dlas_parks(
                uid, catalog.raw_unique_ids, catalog.raw_dla_confidences, catalog.raw_z_dlas, catalog.raw_log_nhis, 
                dla_confidence, min_z_dla, min_log_nhi)

            uid_matrix[i, 0] = uid
            uid_matrix[i, 1] = n 
            uid_matrix[i, 2] = m

            if m >= size:
                m = size - 1 # max size fix to Garnett's size
            confusion_matrix[n, m] += 1


        # hard-cut
        return confusion_matrix, uid_matrix


    @staticmethod
    def prediction_json2dict(filename_parks="predictions_DR12.json", object_name='dlas'):
        '''
        extract dlas or subdlas or lyb 
        and convert to a dataframe

        Parameters : 
        --- 
        filename_parks (str) : predictions_DR12.json, from Parks (2018)
        object_name (str) : "dlas" or "subdlas" or "lyb"

        Return : 
        ---
        dict_parks (dict) : a dictionary contains the predictions from Parks (2018)
        '''
        import json
        with open(filename_parks, 'r') as f:
            parks_json = json.load(f)

        num_dlas = "num_{}".format(object_name)

        # extract DLA information (exclude subDLA, lyb)
        ras       = []
        decs      = []
        plates    = []
        mjds      = []
        fiber_ids = []
        z_qsos    = []
        dla_confidences = []
        z_dlas    = []
        log_nhis  = []

        for table in parks_json:
            # extract plate-mjd-fiber_id
            plate, mjd, fiber_id = table['id'].split("-")
            
            # has dla(s)
            if table[num_dlas] > 0:
                for i in range(table[num_dlas]):
                    dla_table = table[object_name][i]
                    
                    # append the basic qso info
                    ras.append(table['ra'])
                    decs.append(table['dec'])
                    plates.append(plate)       
                    mjds.append(mjd)           
                    fiber_ids.append(fiber_id) 
                    z_qsos.append(table['z_qso'])
                    
                    # append the object (dla or lyb or subdla) info
                    dla_confidences.append(dla_table['dla_confidence'])
                    z_dlas.append(dla_table['z_dla'])
                    log_nhis.append(dla_table['column_density'])
            
            # no dla
            elif table[num_dlas] == 0:
                # append basic info
                ras.append(table['ra'])
                decs.append(table['dec'])
                plates.append(plate)
                mjds.append(mjd)
                fiber_ids.append(fiber_id)
                z_qsos.append(table['z_qso'])

                # append no dla info
                dla_confidences.append(np.nan)
                z_dlas.append(np.nan)
                log_nhis.append(np.nan)
            
            else:
                print("[Warning] exception case")
                print(table)
                
        dict_parks = {
                'ras'    :          np.array(ras),
                'decs'   :          np.array(decs),
                'plates' :          np.array(plates).astype(np.int),
                'mjds'   :          np.array(mjds).astype(np.int),
                'fiber_ids' :       np.array(fiber_ids).astype(np.int),
                'z_qso'  :          np.array(z_qsos),
                'dla_confidences' : np.array(dla_confidences),
                'z_dlas' :          np.array(z_dlas),
                'log_nhis' :        np.array(log_nhis)
            }

        return dict_parks

    def _get_parks_estimations(self, dla_parks, p_thresh=0.98, prior=False):
        '''
        Get z_dlas and log_nhis from Parks' (2018) estimations
        '''
        if 'dict_parks' not in dir(self):
            self.dict_parks = self.prediction_json2dict(dla_parks)

        if 'p_thresh' in self.dict_parks.keys():
            if self.dict_parks['p_thresh'] == p_thresh:
                unique_ids  = self.dict_parks['unique_ids']
                log_nhis    = self.dict_parks['cddf_log_nhis']  
                z_dlas      = self.dict_parks['cddf_z_dlas']    
                min_z_dlas  = self.dict_parks['min_z_dlas']
                max_z_dlas  = self.dict_parks['max_z_dlas']
                snrs        = self.dict_parks['snrs']      
                all_snrs    = self.dict_parks['all_snrs']  
                p_dlas      = self.dict_parks['cddf_p_dlas']    
                p_thresh    = self.dict_parks['p_thresh']  

                return unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas

        dict_parks = self.dict_parks

        # construct an array of unique ids for los
        self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
        unique_ids      = self.make_unique_id( dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'] ) 
        assert unique_ids.dtype is np.dtype('int64')
        assert self.unique_ids.dtype is np.dtype('int64')

        # fixed range of the sightline ranging from 911A-1215A in rest-frame
        # we should include all sightlines in the dataset        
        roman_inds = np.isin(unique_ids, self.unique_ids)

        z_qsos     = dict_parks['z_qso'][roman_inds]
        uids       = unique_ids[roman_inds]
        
        uids, indices = np.unique( uids, return_index=True )

        # for loop to get snrs from sbird's snrs file
        all_snrs           = np.zeros( uids.shape )

        for i,uid in enumerate(uids):
            real_index = np.where( self.unique_ids == uid )[0][0]

            all_snrs[i]           = self.snrs[real_index]

        z_qsos     = z_qsos[indices]

        min_z_dlas = (1 + z_qsos) *  lyman_limit  / lya_wavelength - 1
        max_z_dlas = (1 + z_qsos) *  lya_wavelength  / lya_wavelength - 1

        # get DLA properties
        # note: the following indices are DLA-only
        dla_inds = dict_parks['dla_confidences'] > 0.005 # use p_thresh=0.005 to filter out non-DLA spectra and 
                                                         # speed up the computation

        unique_ids = unique_ids[dla_inds]
        log_nhis   = dict_parks['log_nhis'][dla_inds]
        z_dlas     = dict_parks['z_dlas'][dla_inds]
        z_qsos     = dict_parks['z_qso'][dla_inds]
        p_dlas     = dict_parks['dla_confidences'][dla_inds]

        # check if all ids are in Roman's sample
        roman_inds = np.isin(unique_ids, self.unique_ids)
        unique_ids = unique_ids[roman_inds]
        log_nhis   = log_nhis[roman_inds]
        z_dlas     = z_dlas[roman_inds]
        z_qsos     = z_qsos[roman_inds]
        p_dlas     = p_dlas[roman_inds]

        # for loop to get snrs from sbird's snrs file
        snrs           = np.zeros( unique_ids.shape )
        log_priors_dla = np.zeros( unique_ids.shape )

        for i,uid in enumerate(unique_ids):
            real_index = np.where( self.unique_ids == uid )[0][0]

            snrs[i]           = self.snrs[real_index]
            log_priors_dla[i] = self.log_priors_dla[real_index]

        # re-calculate dla_confidence based on prior of DLAs given z_qsos
        if prior:
            p_dlas = p_dlas * np.exp(log_priors_dla)
            p_dlas = p_dlas / np.max(p_dlas)

        dla_inds = p_dlas > p_thresh

        unique_ids     = unique_ids[dla_inds]
        log_nhis       = log_nhis[dla_inds]
        z_dlas         = z_dlas[dla_inds]
        z_qsos         = z_qsos[dla_inds]
        p_dlas         = p_dlas[dla_inds]
        snrs           = snrs[dla_inds]
        log_priors_dla = log_priors_dla[dla_inds]

        # get rid of z_dlas larger than z_qsos or lower than lyman limit
        z_cut_inds = (
            z_dlas > ((1 + z_qsos) *  lyman_limit  / lya_wavelength - 1) ) 
        z_cut_inds = np.logical_and(
            z_cut_inds, (z_dlas < ( (1 + z_qsos) *  lya_wavelength  / lya_wavelength - 1 )) )

        unique_ids     = unique_ids[z_cut_inds]
        log_nhis       = log_nhis[z_cut_inds]
        z_dlas         = z_dlas[z_cut_inds]
        z_qsos         = z_qsos[z_cut_inds]
        p_dlas         = p_dlas[z_cut_inds]
        snrs           = snrs[z_cut_inds]
        log_priors_dla = log_priors_dla[z_cut_inds]

        # # for loop to get min z_dlas and max z_dlas search range from processed data
        # min_z_dlas = np.zeros( unique_ids.shape )
        # max_z_dlas = np.zeros( unique_ids.shape )

        # for i,uid in enumerate(unique_ids):
        #     real_index = np.where( self.unique_ids == uid )[0][0]

        #     min_z_dlas[i] = self.min_z_dlas[real_index]
        #     max_z_dlas[i] = self.max_z_dlas[real_index]

        # # Parks chap 3.2: fixed range of the sightline ranging from 900A-1346A in rest-frame
        # min_z_dlas = (1 + z_qsos) *  900   / lya_wavelength - 1
        # max_z_dlas = (1 + z_qsos) *  1346  / lya_wavelength - 1

        # assert np.all( ( z_dlas < max_z_dlas[0] ) & (z_dlas > min_z_dlas[0]) )

        self.dict_parks['unique_ids']    = unique_ids
        self.dict_parks['cddf_log_nhis'] = log_nhis
        self.dict_parks['cddf_z_dlas']   = z_dlas
        self.dict_parks['min_z_dlas']    = min_z_dlas 
        self.dict_parks['max_z_dlas']    = max_z_dlas
        self.dict_parks['snrs']          = snrs
        self.dict_parks['all_snrs']      = all_snrs
        self.dict_parks['cddf_p_dlas']   = p_dlas
        self.dict_parks['p_thresh']      = p_thresh

        return unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas

    def plot_cddf_parks(
            self, dla_parks, zmin=1., zmax=6., label='Parks', color=None, moment=False, 
            p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        plot the column density function of Parks' (2018) catalogue
        '''
        (l_N, cddf, xerrs) = self.column_density_function_parks(
            dla_parks, z_min=zmin, z_max=zmax, p_thresh=p_thresh, 
            snr_thresh=snr_thresh, prior=prior, apply_p_dlas=apply_p_dlas)

        if moment:
            cddf *= 10**l_N

        plt.errorbar(10**l_N, cddf, xerr=(xerrs[0], xerrs[1]), fmt='o', label=label, color=color)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
        plt.ylabel(r"$f(N_\mathrm{HI})$")
        return (l_N, cddf)
        
    def column_density_function_parks(
            self, dla_parks, z_min=1., z_max=6., lnhi_nbins=30, lnhi_min=20., lnhi_max=23., 
            p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        Compute the column density function for Parks' catalogue.
        The column density function if the number of absorbers per 
        sightlines with a given column density interval [NHI, NHI + dNHI]
        at a given absorption distance:

        f(N) = d n_DLA / dN / dX 
        or
        f(N) = n_DLA / ΔN / ΔX where n_DLA is the number of DLA
        in a given bin.

        Parameters:
        ----
        dla_parks (str)  : path to the filename of Parks prediction_DR12.json 
        z_min (float)
        z_max (float)
        lnhi_nbins (int) : number of bins in log NHI
        lnhi_min (float)
        lnhi_max (float)    

        Returns:
        ----
        (l_Ncent, cddf, xerrs)

        Note:
        ----
        See also:
        rmgarnett/gp_dla_detection/CDDF_analysis/calc_cddf.py
        sbird/fake_spectra/spectra.py
        '''
        # log NHI bins
        lnhis = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)

        # get Parks' point estimations
        unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas = self._get_parks_estimations(
            dla_parks, p_thresh=p_thresh, prior=prior)

        # filter based on snr threshold
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs > snr_thresh

        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired samples 
        inds = (log_nhis > lnhi_min) * (log_nhis < lnhi_max) * (z_dlas < z_max) * (z_dlas > z_min)
        inds = np.logical_and( snr_inds, inds )

        log_nhis   = log_nhis[inds]
        z_dlas     = z_dlas[inds]
        p_dlas     = p_dlas[inds]

        # ref: https://github.com/sbird/fake_spectra/blob/master/fake_spectra/spectra.py#L976
        if apply_p_dlas:
            tot_f_N, NHI_table = np.histogram(10**log_nhis, 10**lnhis, weights=p_dlas)
        else:
            tot_f_N, NHI_table = np.histogram(10**log_nhis, 10**lnhis)

        dX = self.path_length(min_z_dlas, max_z_dlas, z_min, z_max)
        dN = np.array( [10**lnhi_x - 10**lnhi_m for (lnhi_m, lnhi_x) in zip( lnhis[:-1], lnhis[1:] )] )

        cddf = tot_f_N / dX / dN

        l_Ncent = np.array([ (lnhi_x + lnhi_m) / 2. for (lnhi_m, lnhi_x) in zip(lnhis[:-1], lnhis[1:]) ])
        xerrs = (10**l_Ncent - 10**lnhis[:-1], 10**lnhis[1:] - 10**l_Ncent)
        
        return (l_Ncent, cddf, xerrs)

    def plot_line_density_park(
            self, dla_parks, zmin=2, zmax=4, label="Parks", color=None, 
            p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        plot the line density as a function of redshift
        '''
        z_cent, dNdX, xerrs = self.line_density_park(
            dla_parks, z_min=zmin, z_max=zmax, 
            p_thresh=p_thresh, snr_thresh=snr_thresh, prior=prior, apply_p_dlas=apply_p_dlas)

        plt.errorbar(z_cent, dNdX, xerr=xerrs, fmt='o', label=label, color=color)
        plt.xlabel(r'z')
        plt.ylabel(r'dN/dX')
        plt.xlim(zmin, zmax)
        return z_cent, dNdX

    def line_density_park(
            self, dla_parks, z_min=2, z_max=4, lnhi_min=20.3, 
            bins_per_z=6, p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        Compute the line density, the total number of DLA slightlines divided by
        the total number of sightlines, multiplied by dL/dX,
        which is dN/dX = l_DLA(z)
        '''
        nbins = np.max([ int( (z_max - z_min) * bins_per_z ), 1])
        
        # get the redshift bins
        z_bins = np.linspace(z_min, z_max, nbins + 1)

        # get Parks' point estimations
        unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas = self._get_parks_estimations(
            dla_parks, p_thresh=p_thresh, prior=prior)

        # filter based on snr threshold
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs > snr_thresh

        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired DLA samples 
        inds = (log_nhis > lnhi_min)
        inds = np.logical_and( snr_inds, inds )

        z_dlas     = z_dlas[inds]
        p_dlas     = p_dlas[inds]

        # point estimate of number of DLAs
        if apply_p_dlas:
            ndlas, _ = np.histogram( z_dlas, z_bins, weights=p_dlas )
        else:
            ndlas, _ = np.histogram( z_dlas, z_bins )

        # calc dX for z_bins
        dX = np.array([ self.path_length(min_z_dlas, max_z_dlas, z_m, z_x) 
            for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ])

        ii = np.where( dX > 0)
        dX = dX[ii]

        dNdX = ndlas[ii] / dX

        z_cent = np.array( [ (z_x + z_m) / 2. for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ] )
        xerrs  = (z_cent[ii] - z_bins[:-1][ii], z_bins[1:][ii] - z_cent[ii])

        return (z_cent[ii], dNdX, xerrs)

    def column_density_function_noterdaeme(
            self, dla_noterdaeme, los_noterdaeme, z_min, z_max, lnhi_nbins=30, lnhi_min=20., lnhi_max=23.,
            snr_thresh=4):
        '''
        Compute the column density distribution funciont for Noterdaeme DR12 catalogue.

        This should follow the convention of sbird's plot

        Note:
        ----
        See self.column_density_function_parks for more science details  
        
        Parameters:
        ----
        dla_noterdaeme (str) : the path to noterdaeme DR12 file. In Roman's setting, it would be
            data/dla_catalogs/dr12q_noterdaeme/processed/dla_catalog
        z_min (float) : the minimum redshift you consider to compute the CDDF
        z_max (float) : the maximum redshift you consdier to compute the CDDF
        lnhi_nbins (int) : the number of bins you put on DLA column densities
        lnhi_min (float) : the minimum log column density of DLAs you consider to plot
        lnhi_max (float) : the maximum log column density of DLAs you consdier to plot
        
        Returns:
        ----
        l_Ncent (np.ndarray) : the array of the centers of the log NHI bins
        cddf (np.ndarray)    : the CDDF you computed, which is f(N) = n_DLA / ΔN / ΔX
        xerrs (np.ndarray)   : the width of each bins you applied on the log NHI bins 
        '''
        # log NHI bins 
        lnhis = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)

        # get Noterdaeme's catalogue values 
        thing_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs  = self._get_noterdaeme_estimations(
            dla_noterdaeme, los_noterdaeme)

        # SNR cut for both all sightlines and DLA slightlines
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs     > snr_thresh

        # update searching ranges from SNR cut
        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired samples
        inds = (log_nhis > lnhi_min) * (log_nhis < lnhi_max) * (z_dlas < z_max) * (z_dlas > z_min)
        
        # also update the snr cuts for the DLA sightlines
        inds = np.logical_and( snr_inds, inds )

        log_nhis = log_nhis[inds]
        z_dlas   = z_dlas[inds]

        # get CDDF from histogram
        tot_f_N, NHI_table = np.histogram(10**log_nhis, 10**lnhis)

        dX = self.path_length(min_z_dlas, max_z_dlas, z_min, z_max)
        dN = np.array( [10**lnhi_x - 10**lnhi_m for (lnhi_m, lnhi_x) in zip( lnhis[:-1], lnhis[1:] ) ] )

        cddf = tot_f_N / dX / dN

        l_Ncent = np.array([ (lnhi_x + lnhi_m) / 2. for (lnhi_m, lnhi_x) in zip(lnhis[:-1], lnhis[1:]) ])
        xerrs   = (10**l_Ncent - 10**lnhis[:-1], 10**lnhis[1:] - 10**l_Ncent)

        return (l_Ncent, cddf, xerrs)

    def plot_cddf_noterdaeme(
            self, dla_noterdaeme, los_noterdaeme, zmin=1., zmax=6., label='Noterdaeme DR12', color=None, moment=False,
            snr_thresh=4):
        '''
        plot the column density function of Noterdaeme DR12 catalogue
        '''
        (l_N, cddf, xerrs) = self.column_density_function_noterdaeme(
            dla_noterdaeme, los_noterdaeme, z_min=zmin, z_max=zmax, snr_thresh=snr_thresh)

        if moment:
            cddf *= 10**l_N

        plt.errorbar(10**l_N, cddf, xerr=(xerrs[0], xerrs[1]), fmt='o', label=label, color=color)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
        plt.ylabel(r"$f(N_\mathrm{HI})$")

        return (l_N, cddf)

    def line_density_noterdaeme(
            self, dla_noterdaeme, los_noterdaeme, z_min=2, z_max=4, lnhi_min=20.3, bins_per_z=6,
            snr_thresh=4):
        '''
        Compute the line density for Noterdaeme DR12
        '''
        nbins = np.max([ int( (z_max - z_min) * bins_per_z ), 1 ])

        # get the redshift bins
        z_bins = np.linspace(z_min, z_max, nbins + 1) 

        # get the Noterdaeme DR12 catalogue values
        thing_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas,snrs,all_snrs = self._get_noterdaeme_estimations(
            dla_noterdaeme, los_noterdaeme)

        # SNR cut for both all sightlines and DLA sightlines
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs     > snr_thresh

        # update the searching ranges using SNR cuts
        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired DLA samples
        inds = (log_nhis > lnhi_min)

        # also update the SNR cuts to the DLA slightlines
        inds = np.logical_and( snr_inds, inds )

        z_dlas = z_dlas[inds]

        # estimate the number of DLAs
        ndlas, _ = np.histogram( z_dlas, z_bins )

        dX = np.array([ self.path_length(min_z_dlas, max_z_dlas, z_m, z_x)
            for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ])

        ii = np.where( dX > 0 )
        dX = dX[ii]

        dNdX = ndlas[ii] / dX

        z_cent = np.array( [ (z_x + z_m) / 2. for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ] )
        xerrs  = (z_cent[ii] - z_bins[:-1][ii], z_bins[1:][ii] - z_cent[ii])

        return (z_cent[ii], dNdX, xerrs)

    def plot_line_density_noterdaeme(
        self, dla_noterdaeme, los_noterdaeme, zmin=2, zmax=4, label="Noterdaeme", color=None,
        snr_thresh=4):
        '''
        plot the line density of Noterdaeme DR12
        '''
        z_cent, dNdX, xerrs = self.line_density_noterdaeme(
            dla_noterdaeme, los_noterdaeme, z_min=zmin, z_max=zmax, snr_thresh=snr_thresh)

        plt.errorbar(z_cent, dNdX, xerr=xerrs, fmt='o', label=label, color=color)
        plt.xlabel(r'z')
        plt.ylabel(r'dN/dX')
        plt.xlim(zmin, zmax)

        return z_cent, dNdX

    def _get_noterdaeme_estimations(self, dla_noterdaeme, los_noterdaeme):
        '''
        Get z_dlas and log_nhis from Noterdaeme DR12 estimations
        
        Returns:
        ----
        thing_ids  : DLA thing_ids (could have more than one DLAs)
        log_nhis   : DLA log NHI
        z_dlas     : DLA z_dlas
        min_z_dlas : minimum search absorber redshift for a given sightline
        max_z_dlas : maximum search absorber redshift for a given sightline
        all_snrs   : signal-to-noise ratios for all sightlines
        snrs       : signal-to_noise ratios for DLA sightlines (could have multiple DLAs)

        Note: only consider sightlines overlapping with Garnett's
        '''
        # Get the search length first, searching length dX should consider all of the sightlines
        # All slightlines:
        all_thing_ids = np.loadtxt(los_noterdaeme).astype(np.int)

        # check if these are in our catalogue
        inds      = np.isin(self.thing_ids, all_thing_ids)
        z_qsos    = self.z_qsos[inds]
        all_snrs  = self.snrs[inds]

        # Noterdaeme 2012 section 2.2 (DOI: 10.1051/0004-6361/201220259)
        # They consider only the region between 3000 km s−1 redwards of
        # the Ly-β emission line and 5000 km s−1 bluewards of the Ly-α emission line
        min_z_dlas = (1 + z_qsos) * (lyb_wavelength + kms_to_z(3000)) / lya_wavelength - 1
        max_z_dlas = (1 + z_qsos) * (lya_wavelength - kms_to_z(5000)) / lya_wavelength - 1

        # DLA slightlines:
        catalog = np.loadtxt(dla_noterdaeme)

        thing_ids = catalog[:, 0].astype(np.int)
        z_dlas    = catalog[:, 1]
        log_nhis  = catalog[:, 2]

        roman_inds = np.isin(thing_ids, self.thing_ids)

        thing_ids = thing_ids[roman_inds]
        z_dlas    = z_dlas[roman_inds]
        log_nhis  = log_nhis[roman_inds]

        # search for snrs of DLA sightlines
        index = search_index_from_another(thing_ids, self.thing_ids)
        assert np.all( thing_ids == self.thing_ids[index] )        

        snrs = self.snrs[index]

        return thing_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs


    @staticmethod
    def path_length(min_z_dlas, max_z_dlas, z_min, z_max):
        '''
        Compute the path length, dX, over which we looked for DLAs.

        dz = max_z_dlas - min_z_dlas

        dX = (1 + z)^2 / H_0 / H(z) dz

        Reference: calc_cddf.DLACatalogue.path_length
        '''
        total_dX = 0

        # filter the spectra aren't in our range
        inds = np.logical_and( min_z_dlas < z_max, max_z_dlas > z_min )
        min_z_dlas = min_z_dlas[inds]
        max_z_dlas = max_z_dlas[inds]

        # for spectra across the whole dz bin
        whole_dz_bin = np.logical_and( max_z_dlas > z_max, min_z_dlas < z_min )

        (tbin, err) = integrate.quad( path_length_int, z_min, z_max )
        total_dX += np.size( np.where(whole_dz_bin) ) * tbin

        inds = np.logical_not( whole_dz_bin )
        max_z_dlas = max_z_dlas[inds]
        min_z_dlas = min_z_dlas[inds]

        for (zmin, zmax) in zip(min_z_dlas, max_z_dlas):
            assert zmin <= zmax

            pathzmin = np.max([ z_min, zmin ])
            pathzmax = np.min([ z_max, zmax ])

            (ans, err) = integrate.quad(path_length_int, pathzmin, pathzmax)
            total_dX += ans
            assert err < 1e-6
        
        return total_dX


    def find_this_flux(self, nspec):
        '''
        Find the flux measurement of this QSO spectrum

        Parameters:
        ----
        nspec (int) : the index in the test data (processed data)
        '''
        nspec_real = self.test_real_index[nspec]

        return self.preloaded_file[ self.preloaded_file['all_flux'][0, nspec_real] ][0]

    def find_this_wavelengths(self, nspec):
        '''
        Find the observed wavelengths of this QSO spectrum

        Parameters:
        ----
        nspec (int) : the index in the test data (processed data)
        '''
        nspec_real = self.test_real_index[nspec]

        return self.preloaded_file[ self.preloaded_file['all_wavelengths'][0, nspec_real] ][0]

    def find_this_noise_variance(self, nspec):
        '''
        Find the noise variance per pixel of this QSO spectrum

        Paramters:
        ----
        nspec (int) : the index in the test data (processed data)
        '''
        nspec_real = self.test_real_index[nspec]

        return self.preloaded_file[ self.preloaded_file['all_noise_variance'][0, nspec_real] ][0]

    
    def plot_mean_flux(self, nspec, suppressed=True, num_lines=31):
        '''
        Plot mean-flux with observed flux
        '''
        this_wavelengths = self.find_this_wavelengths(nspec)
        this_flux        = self.find_this_flux(nspec)

        this_rest_wavelengths = emitted_wavelengths(this_wavelengths, self.z_qsos[nspec])
        
        this_mu = self.GP.mu
         
        if suppressed:
            # count the effective optical depth from members in Lyman series
            scale_factor = self.total_scale_factor(
                self.GP.tau_0_kim, self.GP.beta_kim, self.z_qsos[nspec], self.GP.rest_wavelengths, num_lines=num_lines)
            this_mu = this_mu * scale_factor

        plt.figure(figsize=(16, 5))
        plt.plot(this_rest_wavelengths, this_flux, label=r"observed flux", color="C0")
        plt.plot(self.GP.rest_wavelengths, this_mu, label=r"mean-flux $\mu \circ exp(-\tau (1+z)^\beta)$", color="red")
        plt.xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
        plt.ylabel(r"normalised flux")
        plt.legend()
        return self.GP.rest_wavelengths, this_mu

    def plot_this_mu(self, nspec, suppressed=True, num_voigt_lines=3, num_forest_lines=31, Parks=False, dla_parks=None, 
        label="", new_fig=True, color="red"):
        '''
        Plot the spectrum with the dla model

        Parameters:
        ----
        nspec (int) : index of the spectrum in the catalogue
        suppressed (bool) : apply Lyman series suppression to the mean-flux or not
        num_voigt_lines (int, min=1, max=31) : how many members of Lyman series in the DLA Voigt profile
        number_forest_lines (int) : how many members of Lymans series considered in the froest
        Parks (bool) : whether to plot Parks' results
        dla_parks (str) : if Parks=True, specify the path to Parks' `prediction_DR12.json`

        Returns:
        ----
        rest_wavelengths : rest wavelengths for the DLA model
        this_mu : the DLA model
        map_z_dlas : MAP z_dla values 
        map_log_nhis : MAP log NHI values
        '''
        # spec id
        plate, mjd, fiber_id = (self.plates[nspec], self.mjds[nspec], self.fiber_ids[nspec])

        # for obs data
        this_wavelengths = self.find_this_wavelengths(nspec)
        this_flux        = self.find_this_flux(nspec)

        this_rest_wavelengths = emitted_wavelengths(this_wavelengths, self.z_qsos[nspec])

        # for building GP model
        rest_wavelengths = self.GP.rest_wavelengths
        this_mu = self.GP.mu

        # count the effective optical depth from members in Lyman series
        scale_factor = self.total_scale_factor(
            self.GP.tau_0_kim, self.GP.beta_kim, self.z_qsos[nspec], self.GP.rest_wavelengths, num_lines=num_forest_lines)
        if suppressed:
            this_mu = this_mu * scale_factor

        # get the MAP DLA values
        nth = np.argmax( self.model_posteriors[nspec] ) - 1 - self.sub_dla
        if nth >= 0:
            if self.model_posteriors.shape[1] > 2:
                map_z_dlas    = self.all_z_dlas[nspec, :(nth + 1)]
                map_log_nhis  = self.all_log_nhis[nspec, :(nth + 1)]
            elif self.model_posteriors.shape[1] == 2: # Garnett (2017) model
                self.prepare_roam_map_vals_per_spec(nspec, self.sample_file)
                
                map_z_dlas    = np.array([ self.all_z_dlas[nspec] ])
                map_log_nhis  = np.array([ self.all_log_nhis[nspec] ])
                assert ~np.isnan(map_z_dlas)

            for map_z_dla, map_log_nhi in zip(map_z_dlas, map_log_nhis):
                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]), 10**map_log_nhi, map_z_dla, num_lines=num_voigt_lines)

                this_mu = this_mu * absorption

        # get parks model
        if Parks:
            if not 'dict_parks' in dir(self):
                self.dict_parks = self.prediction_json2dict(dla_parks)

            dict_parks = self.dict_parks

            # construct an array of unique ids for los
            self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
            unique_ids      = self.make_unique_id( dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'])  
            assert unique_ids.dtype is np.dtype('int64')
            assert self.unique_ids.dtype is np.dtype('int64')

            uids = np.where( unique_ids == self.unique_ids[nspec] )[0]

            this_parks_mu = self.GP.mu * scale_factor
            dla_confidences = []
            z_dlas          = []
            log_nhis        = []
            
            for uid in uids:
                z_dla   = dict_parks['z_dlas'][uid]
                log_nhi = dict_parks['log_nhis'][uid]

                dla_confidences.append( dict_parks['dla_confidences'][uid] )
                z_dlas.append( z_dla )
                log_nhis.append( log_nhi )

                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]), 10**log_nhi, z_dla, num_lines=1)

                this_parks_mu = this_parks_mu * absorption

        # plt.figure(figsize=(16, 5))
        if new_fig:
            make_fig()
            plt.plot(this_rest_wavelengths, this_flux, label="observed flux; spec-{}-{}-{}".format(plate, mjd, fiber_id), color="C0")

        if Parks:
            plt.plot(rest_wavelengths, this_parks_mu, label=r"Parks: z_dlas = ({}); lognhis=({}); p_dlas=({})".format(
                ",".join("{:.3g}".format(z) for z in z_dlas), 
                ",".join("{:.3g}".format(n) for n in log_nhis), 
                ",".join("{:.3g}".format(p) for p in  dla_confidences)), 
                color="orange")
        if nth >= 0:
            plt.plot(rest_wavelengths, this_mu, 
                label=label + r"$\mathcal{M}$"+r" DLA({n})".format(n=nth+1) + ": {:.3g}; ".format(
                    self.model_posteriors[nspec, 1 + self.sub_dla + nth]) + 
                    "lognhi = ({})".format( ",".join("{:.3g}".format(n) for n in map_log_nhis) ), 
                color=color)
        else:
            plt.plot(rest_wavelengths, this_mu, 
                label=label + r"$\mathcal{M}$"+r" DLA({n})".format(n=0) + ": {:.3g}".format(self.p_no_dlas[nspec]), 
                color=color)

        plt.xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
        plt.ylabel(r"normalised flux")
        plt.legend()
        
        if nth >= 0:
            return rest_wavelengths, this_mu, map_z_dlas, map_log_nhis            
        return rest_wavelengths, this_mu

    @staticmethod
    def total_scale_factor(tau, beta, z_qso, rest_wavelengths, num_lines=31):
        '''
        Calculate total absorption effect by scaling the effective optical depth
        from Kim (2019).

        scale_factor = exp( - tau (1 + z)^beta )
                     = exp( - effective_optical_depth )

        Parameters:
        ----
        tau (float)  : prior of optical depth from Kim (2007) for HI
        beta (float)
        z_qso (float)
        rest_wavelengths (np.ndarray)
        num_lines (int) : number of members in Lyman series are used

        Returns:
        ---- 
        scale_factor (float) : exp( - (total_effective_optical_depth)^beta )
        '''
        total_optical_depth    = np.empty( ( num_lines,  len(rest_wavelengths)) )
        total_optical_depth[:] = np.nan

        for i in range(num_lines):
            # calculate the effective optical depth for each member of Lyman series
            this_lyman_1pzs = rest_wavelengths * (1 + z_qso) / all_transition_wavelengths[i]

            # multiply the indicator function I(zmin, zqso)(z)
            indicator = (this_lyman_1pzs <= (1 + z_qso)).astype(np.float)

            # avoid the z_qso != max(z_lya) problem
            if i != 0:
                this_lyman_1pzs = this_lyman_1pzs * indicator

            # scale the optical depth from Kim (2007) prior
            this_tau = tau_lyseries(
                tau, all_oscillator_strengths[i], all_transition_wavelengths[i])

            total_optical_depth[i, :] = this_tau * this_lyman_1pzs**beta

        # scale_factor = exp( - effective_optical_depth )        
        scale_factor = np.exp( - np.nansum( total_optical_depth, axis=0 ) )

        assert scale_factor.shape[0] == rest_wavelengths.shape[0] 

        return scale_factor

    def plot_raw_spectrum(self, nspec, release='dr12q', download=True):
        '''
        Plot the raw spectrum, the spectrum before normalisation.

        if download=True, try to download the raw spectrum from the SDSS website.
        '''
        filename = file_loader(
            release, self.plates[nspec], self.mjds[nspec], self.fiber_ids[nspec])

        if not os.path.exists(filename):
            dirname = "{}/{:d}".format(
                    spectra_directory(release), self.plates[nspec])
            if not os.path.isdir(dirname):
                os.mkdir(dirname)

            # download raw data from sdss website
            self.retrieve_raw_spec(
                self.plates[nspec], self.mjds[nspec], self.fiber_ids[nspec], release=release)

        # read fits file
        wavelengths, flux, noise_variance, pixel_mask = self.read_spec(filename)

        # plotting config
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(wavelengths, flux,           lw=0.25, label=r"$y(\lambda_{obs})$")
        ax.plot(wavelengths, noise_variance, lw=0.25, label=r"$\sigma^2(\lambda_{obs})$")
        ax.set_xlabel(r"observed wavelength [$\AA$]")
        ax.set_ylabel(r"flux [$10^{-17}{erg}/s/{cm}^{2}/\AA$]")
        ax.set_title(r"{} at z_qso = {:.3g}".format(
            filename, self.z_qsos[nspec]))
        ax.set_ylim( np.quantile(flux, 0.005), np.quantile(flux, 0.995) )
        ax.legend()

        ax2 = ax.secondary_xaxis('top', functions=(
            lambda x : emitted_wavelengths(x, self.z_qsos[nspec]), 
            lambda x : observed_wavelengths(x, self.z_qsos[nspec])))
        ax2.set_xlabel(r"rest wavelength [$\AA$]")
        
        return wavelengths, flux

    @staticmethod
    def read_spec(filename):
        '''
        python version of Roman's read_spec.m

        Returns:
        ----
        wavelengths     : observed wavelengths 
        flux            : coadded calibrated flux 10**-17 erg s**-1 cm**-2 A**-1
        noise_variance  : noise variance per pixel
        pixel_mask      : if 1/noise_variance = 0 and BRIGHTSKY
        '''
        from astropy.io import fits
        with fits.open(filename) as hdu:
            data = hdu['COADD'].data

            # coadded calibrated flux 10**-17 erg s**-1 cm**-2 A**-1
            flux                   = data['flux']

            # log_10 wavelength       log (A)
            log_wavelengths        = data['loglam']

            # inverse noise variance
            inverse_noise_variance = data['ivar']

            # `and` mask
            and_mask               = data['and_mask'] 

            wavelengths    = 10**log_wavelengths
            noise_variance = 1 / inverse_noise_variance

            # derive bad pixel mask, follow the same recipe in Roman's read_spec.m
            BRIGHTSKY = 24
            pixel_mask = (inverse_noise_variance == 0) | np.array(
                [(m >> BRIGHTSKY) & 1 for m in and_mask]).astype("bool")

        return wavelengths, flux, noise_variance, pixel_mask

    @staticmethod
    def retrieve_raw_spec(plate, mjd, fiber_id, release='dr12q'):
        '''
        download raw spec
        '''
        filename = file_loader(release, plate, mjd, fiber_id)

        # greedy list all plates at v_5_7_2
        v_5_7_2_plates = [7339, 7340, 7386, 7388, 7389, 7391, 7396, 7398, 7401, 
                        7402, 7404, 7406, 7407, 7408, 7409, 7411, 7413, 7416, 
                        7419, 7422, 7425, 7426, 7428, 7455, 7512, 7513, 7515, 
                        7516, 7517, 7562, 7563, 7564, 7565]

        from urllib import request

        in_5_7_2 = plate in v_5_7_2_plates
        url = 'https://data.sdss.org/sas/dr12/boss/spectro/redux/{}/spectra/{:d}/spec-{:d}-{:d}-{:04d}.fits'.format(
            ['v5_7_0', 'v5_7_2'][in_5_7_2], plate, plate, mjd, fiber_id
        )

        print("[Info] retrieving {} ...".format(url), end=" ")
        request.urlretrieve(url, filename)
        print("Done.")


    def generate_json_catalogue(self, outfile="predictions_multi_DLAs.json"):
        '''
        Generate the Multi-DLA catalogue in JSON format, 
        which has the same form as Parks' (2018) product.

        Example template :
        ----
        [
            {
                "p_dla" : 0.9,
                "p_no_dla" : 0.1,
                "max_model_posterior": 0.7,
                "num_dlas": 1, 
                "dlas": [
                    {
                        "log_nhi": 20.78598574580358, 
                        "z_dla": 2.2291207038222756
                    }
                ], 
                "min_z_dla": 2.0,
                "max_z_dla": 2.18,
                "ra": 9.2152, 
                "snr": 1.23,
                "dec": -0.1659, 
                "plate": 3586,
                "mjd": 55181,
                "fiber_id": 16,
                "thing_id": ,
                "z_qso": 2.190
            }, 
            ...
        ]
        '''
        # initialise a list to store dicts
        # we will save each spectrum into a dict.
        # if there's any dla, we will just save the dlas as a list of 
        # dicts as an item in the spectrum dict.
        predictions_DLAs = []

        # query the maximum values of the model posteriors per spectrum first
        model_index = self.model_posteriors.argmax(axis=1)
        max_model_posteriors = self.model_posteriors.max(axis=1)

        if self.sub_dla:
            # now we need to combine sub-DLAs with null model posterior
            # to get the mosterior of no-DLAs
            inds = model_index < (1 + self.sub_dla)
            max_model_posteriors[inds] = self.p_no_dlas[inds]

            # prepare to use model_index as num_dlas
            num_dlas = model_index - self.sub_dla
            num_dlas[num_dlas < 0] = 0 # num_dlas should be positive

        assert len(max_model_posteriors) == len(self.thing_ids)

        # store some arrays here to query later
        # you do the test_ind indicing after storing the HDF5 array 
        # into memory. Reading from numpy array from memory is much
        # faster than you do IO from the file.
        ras  = self.catalogue_file['ras'][0, :][self.test_ind]
        decs = self.catalogue_file['decs'][0, :][self.test_ind]

        assert len(ras) == len(self.thing_ids)

        for i,thing_id in enumerate(self.thing_ids):
            spec = dict()

            # you need to put .item() to convert numpy datatype to python datatype
            # since json doesn't support serialize numpy datatype
            spec["p_dla"]               = self.p_dlas[i].item()
            spec["p_no_dla"]            = self.p_no_dlas[i].item()
            spec["max_model_posterior"] = max_model_posteriors[i].item()
            spec["num_dlas"]            = num_dlas[i].item()
            spec["min_z_dla"]           = self.min_z_dlas[i].item()
            spec["max_z_dla"]           = self.max_z_dlas[i].item()
            spec["snr"]                 = self.snrs[i].item()
            spec["ra"]                  = ras[i].item()
            spec["dec"]                 = decs[i].item()
            spec["plate"]               = self.plates[i].item()
            spec["mjd"]                 = self.mjds[i].item()
            spec["fiber_id"]            = self.fiber_ids[i].item()
            spec["thing_id"]            = thing_id.item()
            spec["z_qso"]               = self.z_qsos[i].item()

            dlas = []

            if num_dlas[i] > 0:
                this_map_z_dlas   = self.map_z_dlas[i, (num_dlas[i] - 1), :num_dlas[i]]
                this_map_log_nhis = self.map_log_nhis[i, (num_dlas[i] - 1), :num_dlas[i]]

                assert len(this_map_z_dlas) == num_dlas[i]

                for j in range(num_dlas[i]):
                    dlas.append({
                        "log_nhi" : this_map_log_nhis[j],
                        "z_dla"   : this_map_z_dlas[j]
                    })
            
            spec["dlas"] = dlas

            predictions_DLAs.append(spec)

        import json
        with open(outfile, "w") as json_file:
            json.dump(predictions_DLAs, json_file, indent=2)

        return predictions_DLAs

    def generate_sub_dla_catalogue(self, outfile="predictions_sub_DLA_candidates.json"):
        '''
        Generate a catalogue for the candidates of sub-DLAs:
        we only record the sub-DLA probability and the QSO spec info for this catalogue
        
        Example template :
        ----
        [
            {
                "p_sub_dla" : 0.9,
                "ra": 9.2152, 
                "snr": 1.23,
                "dec": -0.1659, 
                "plate": 3586,
                "mjd": 55181,
                "fiber_id": 16,
                "thing_id": ,
                "z_qso": 2.190
            }, 
            ...
        ]        
        '''
        # repeat the same process in self.gerenate_json_catalogue()
        # but only consider the specs with sub-DLA has the highest prob
        predictions_sub = []

        # query the quasar_ind of sub-DLA model posterior is the highest
        quasar_inds = np.where(self.dla_map_model_index == 1)[0]

        # store some arrays here to query later
        # you do the test_ind indicing after storing the HDF5 array 
        # into memory. Reading from numpy array from memory is much
        # faster than you do IO from the file.
        ras  = self.catalogue_file['ras'][0, :][self.test_ind]
        decs = self.catalogue_file['decs'][0, :][self.test_ind]

        for i in quasar_inds:
            this_sub_dla = {}

            this_sub_dla['p_sub_dla'] = self.model_posteriors[i, 1].item()
            this_sub_dla['ra']        = ras[i].item()
            this_sub_dla['snr']       = self.snrs[i].item()
            this_sub_dla['dec']       = decs[i].item()
            this_sub_dla['plate']     = self.plates[i].item()
            this_sub_dla['mjd']       = self.mjds[i].item()
            this_sub_dla['fiber_id']  = self.fiber_ids[i].item()
            this_sub_dla['thing_id']  = self.thing_ids[i].item()
            this_sub_dla['z_qso']     = self.z_qsos[i].item()

            predictions_sub.append(this_sub_dla)

        import json
        with open(outfile, 'w') as json_file:
            json.dump(predictions_sub, json_file, indent=2)
        
        return predictions_sub


file_loader = lambda release, plate, mjd, fiber_id : "{}/{:d}/spec-{:d}-{:d}-{:04d}.fits".format(
    spectra_directory(release), plate, plate, mjd, fiber_id)

def search_index_from_another(y, target):
    '''
    Parameters:
    ----
    y : input array
    target : target array which you want to find indices from
    
    Return:
    ----
    result : masked array of indicies maping from y -> target
        you do target[result] to get corresponding mapped results 
    '''
    sorted_index = np.searchsorted(target[np.argsort(target)], y)

    y_index = np.take(np.argsort(target), sorted_index, mode="clip")
    mask = target[y_index] != y

    return np.ma.array(y_index, mask=mask)

def make_fig():
    '''this is the size of a spectrum'''
    plt.figure(figsize=(16, 5))
    