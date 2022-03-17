# -*- coding: utf-8 -*-

import GPy
import GPyOpt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import numpy as np

import torch.nn as nn
import torch
from torch.distributions import Normal, kl_divergence, Multinomial, Binomial, Poisson

from ._vae import VAE
from ._loss_functions import loss_fn
from tqdm import tqdm

#########################################################################
### Objective function for Hyperparams Optimization using GpyOpt
#########################################################################

def obj_func(native_signals_pred, probs, scRNAseq_tech, empty_frequencies):
    
    if scRNAseq_tech.lower() == 'cropseq':
        native_clean = (probs == probs.max(axis=1).reshape(-1,1))
    else:
        native_clean = native_signals_pred #.cpu().numpy()
        
    frac_native = native_clean.sum(axis=0)/native_clean.sum()
    freq_truth = empty_frequencies.detach().cpu().numpy()
    
    # metric to evaluate the predection
    # mae = median_absolute_error(freq_truth, frac_native)
    # pr = pearsonr(freq_truth, frac_native)[0]
    r2 = r2_score(freq_truth, frac_native)

    return r2 #mae, pr

#########################################################################
### 
#########################################################################

def training_model_with_a_set_params(train_set, total_set, num_input_feature=99,
                                     NN_layer1=200,
                                     NN_layer2=50,
                                     latent_space=10,
                                     kld_weight=0,
                                     frac_match_weight=0,
                                     dropout_prob=0,
                                     lr=0.002,
                                     reconstruction_weight=1,
                                     scRNAseq_tech='CROPseq',
                                     lr_step_size=5,
                                     lr_gamma=0.97,
                                     epochs = 50):

    '''
    train the model with a set of parameters
    '''
        
    # Define model
    model = VAE(num_input_feature, NN_layer1, NN_layer2, latent_space, scRNAseq_tech=scRNAseq_tech).cuda()

    # Define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_step_size, gamma=lr_step_size)

    print("......kld_weight: ", kld_weight)
    print("......lr: ", lr)
    print("......lr_step_size: ", lr_step_size)
    print("......lr_gamma: ", lr_gamma)
    # Training
    for epoch in tqdm(range(epochs)):

        model.train()
        for x_batch, ambient_freq in train_set:
            optim.zero_grad()
            z, dec_nr, dec_prob,  mu, var = model(x_batch)
            recon_loss_minibatch, kld_loss_minibatch, match_loss_minibatch, loss_minibatch = loss_fn(x_batch, dec_nr, dec_prob, mu, var, ambient_freq,
                                                                                         reconstruction_weight=reconstruction_weight, kld_weight=kld_weight, frac_match_weight=frac_match_weight)
            loss_minibatch.backward()
            optim.step()
        scheduler.step()

    # evaluation
    model.eval()
    with torch.no_grad():
        expected_native_counts, probs = model.inference(total_set, ambient_freq[0,:])
    
    r2 = obj_func(expected_native_counts, probs, scRNAseq_tech, empty_frequencies=ambient_freq[0,:])
    
    return r2


############################################################################################################
### Bayesian optimization of hyperparameters
############################################################################################################

class ParamsOpt():
    '''
    Use Bayesian Optimization for hyperparams optimization
    '''
    def __init__(self, train_set, total_set, num_input_feature, scRNAseq_tech, epochs=150, initial_design_numdata=30, max_iter=50, lr_step_size=5, lr_gamma=0.97):
        self.train_set = train_set
        self.total_set = total_set
        self.num_input_feature = num_input_feature
        self.scRNAseq_tech = scRNAseq_tech
        self.epochs = epochs
        self.initial_design_numdata = initial_design_numdata
        self.max_iter = max_iter
#         self.lr = lr   lr=0.002, 
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.domain = [{'name': 'NN_layer1',      'type': 'continuous', 'domain': (np.log(50), np.log(250))},     # interval of NN_layer1 is [10, 1000]
                       {'name': 'NN_layer2',      'type': 'continuous', 'domain': (np.log(50), np.log(250))},     # interval of NN_layer2 is [10, 1000]
                       {'name': 'latent_space',   'type': 'continuous', 'domain': (np.log(5), np.log(20))},       # interval of latent_space is [10, 100]
                       {'name': 'kld_weight',     'type': 'continuous', 'domain': (np.log(1e-6), np.log(1e-2))},  # interval of KLD weight is [1e-6, 1e-2]
                       {'name': 'lr',             'type': 'continuous', 'domain': (np.log(1e-3), np.log(1e-2))}]  # interval of KLD weight is [1e-6, 1e-2] 
    
    def BayesianOptimization(self):

        def hyperparams_opt(domain):
            params = np.atleast_2d(np.exp(domain))
            fs = np.zeros((params.shape[0],1))
            for i in range(params.shape[0]):
                try:
                    fs[i] = training_model_with_a_set_params(self.train_set,
                                                                self.total_set,
                                                                num_input_feature=self.num_input_feature,    ## num of feature barcodes
                                                                NN_layer1=int(params[i, 0]),
                                                                NN_layer2=int(params[i, 1]),
                                                                latent_space=int(params[i, 2]),
                                                                kld_weight=params[i, 3],
                                                                lr=params[i, 4],
                                                                epochs = self.epochs,
                                                                reconstruction_weight=1,
                                                                scRNAseq_tech=self.scRNAseq_tech,
                                                                lr_step_size=self.lr_step_size,
                                                                lr_gamma=self.lr_gamma)
                    
                ## if the model doesn't succeed, return a very low R2_score value.
                except:
                    fs[i] = -1
                    
            return fs

        opt = GPyOpt.methods.BayesianOptimization(f = hyperparams_opt,        # function to optimize       
                                          domain = self.domain,         # box-constraints of the problem
                                          initial_design_numdata = self.initial_design_numdata, 
                                          acquisition_type ='LCB',       # LCB acquisition
                                          maximize = True)

        opt.run_optimization(max_iter=self.max_iter) #eps = self.tolerance
        x_best = np.exp(opt.X[np.argmin(opt.Y)])
        best_params = dict(zip([el['name'] for el in self.domain], x_best))
        self.opt = opt
        self.best_params = best_params