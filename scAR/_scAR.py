# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, sys, time
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union
from sklearn.model_selection import train_test_split
from ._vae import VAE
from ._loss_functions import loss_fn
from ._helper_functions import (histgram_noise_ratio,  get_correlation_btn_native_ambient,
                              plt_correlation_btn_native_ambient, assignment_accuracy,
                              naive_assignment_accuracy, plot_a_sample)

import contextlib
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

# Writing progressbar into stdout rather than stderr, from https://github.com/tqdm/tqdm/blob/master/examples/redirect_print.py
@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

# scAR object
class model():
    
    """
    single cell Ambient Remover [Sheng2022].
    Parameters
    ----------
    ...
    """

    def __init__(self,
                 raw_count: Union[str, np.ndarray],
                 empty_profile: Optional[Union[str, np.ndarray]] = None,
                 NN_layer1: int=150,
                 NN_layer2: int=100,
                 latent_space: int=15,
                 scRNAseq_tech: str="scRNAseq",
                ):

        if isinstance(raw_count, str):
            raw_count = pd.read_pickle(raw_count)
            raw_count = raw_count.fillna(0).values # replace missing values with zeros
        elif isinstance(raw_count, np.ndarray):
            pass
        else:
            raise TypeError("Expecting str or np.array object, but get a {}".format(type(raw_count)))

        if isinstance(empty_profile, str):
            empty_profile = pd.read_pickle(empty_profile)
            empty_profile = empty_profile.fillna(0).values # replace missing values with zeros
        elif isinstance(empty_profile, np.ndarray):
            pass
        elif not empty_profile:
            print(' ... Evaluate empty profile from cells')
            empty_profile = np.sum(raw_count, axis=0)/np.sum(raw_count)       
        else:
            raise TypeError("Expecting str / np.array / None, but get a {}".format(type(empty_profile)))
        
        if empty_profile.squeeze().ndim == 1:
            empty_profile = empty_profile.squeeze().reshape(1,-1).repeat(raw_count.shape[0], axis=0)
        
        self.num_input_feature = raw_count.shape[1]
        self.NN_layer1 = NN_layer1
        self.NN_layer2 = NN_layer2
        self.latent_space = latent_space
        self.scRNAseq_tech = scRNAseq_tech
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading numpy to tensor on GPU
        self.raw_count = torch.from_numpy(raw_count).int().to(self.device)
        self.empty_profile = torch.from_numpy(empty_profile).float().to(self.device)
        
    def train(self,
              batch_size: int=64,
              train_size: float=0.998,
              shuffle: bool=True,
              kld_weight: float=1e-5,
              lr: float=1e-3,
              lr_step_size: int=5,
              lr_gamma: float=0.97,
              epochs: int=800,
              reconstruction_weight: float=1,
              dropout_prob: float=0,
              plot_every_epoch: int=50,
              TensorBoard: bool=False,
              save_model: bool=False):
        
        list_IDs = list(range(self.raw_count.shape[0]))
        train_IDs, test_IDs = train_test_split(list_IDs, train_size=train_size)
        
        # Generators
        training_set = UMIDataset(self.raw_count, self.empty_profile, train_IDs)
        training_generator = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=shuffle)

        val_set = UMIDataset(self.raw_count, self.empty_profile, test_IDs)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle)

        self.n_batch_train = len(training_generator)
        self.n_batch_val = len(val_generator)
        self.batch_size = batch_size
        
        # TensorBoard writer
        if TensorBoard:
            writer = SummaryWriter(TensorBoard)
            writer.add_text('Experiment description',
                            f'NN_layer1={self.NN_layer1}, NN_layer2={self.NN_layer2}, latent_space={self.latent_space}, kld_weight={kld_weight}, lr={lr}, epochs={epochs}, reconstruction_weight={reconstruction_weight}, dropout_prob={dropout_prob}', 0)

        # Define model
        VAE_model = VAE(self.num_input_feature, self.NN_layer1, self.NN_layer2, self.latent_space, self.scRNAseq_tech, dropout_prob).cuda()

        # Define optimizer
        optim = torch.optim.Adam(VAE_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_step_size, gamma=lr_gamma)

        print("......kld_weight: ", kld_weight)
        print("......lr: ", lr)
        print("......lr_step_size: ", lr_step_size)
        print("......lr_gamma: ", lr_gamma)

        # Run training
        print('===========================================\n  Training.....')
        training_start_time = time.time()
        with std_out_err_redirect_tqdm() as orig_stdout:
                        
            for epoch in tqdm(range(epochs), file=orig_stdout, dynamic_ncols=True): # tqdm needs the original stdout and dynamic_ncols=True to autodetect console width

                ################################################################################
                # Training
                ################################################################################
                train_tot_loss = 0
                train_kld_loss = 0
                train_recon_loss = 0

                VAE_model.train()
                for x_batch, ambient_freq in training_generator:
                    
                    optim.zero_grad()
                    z, dec_nr, dec_prob,  mu, var = VAE_model(x_batch)
                    recon_loss_minibatch, kld_loss_minibatch, loss_minibatch = loss_fn(x_batch, dec_nr, dec_prob, mu, var, ambient_freq, reconstruction_weight=reconstruction_weight, kld_weight=kld_weight)
                    loss_minibatch.backward()
                    optim.step()

                    train_tot_loss += loss_minibatch.detach().item()
                    train_recon_loss += recon_loss_minibatch.detach().item()
                    train_kld_loss += kld_loss_minibatch.detach().item()
                
                scheduler.step()

                # ...log the running training loss
                if TensorBoard:
                    writer.add_scalar('TrainingLoss/total loss', train_tot_loss, epoch)
                    writer.add_scalar('TrainingLoss/reconstruction loss', train_recon_loss, epoch)
                    writer.add_scalar('TrainingLoss/kld_loss', train_kld_loss, epoch)
                    writer.add_scalar('learning rate', optim.param_groups[0]['lr'], epoch)
                    # writer.flush()

                # ...log the running validation loss
                if TensorBoard:
                    ###################################################################################
                    # model evaluation
                    ###################################################################################
                    val_tot_loss = 0
                    val_kld_loss = 0
                    val_recon_loss = 0

                    VAE_model.eval()
                    for x_batch_val, ambient_freq_val in val_generator:
                                               
                        z_val, dec_nr_val, dec_prob_val,  mu_val, var_val = VAE_model(x_batch_val)
                        recon_loss_minibatch, kld_loss_minibatch, loss_minibatch = loss_fn(x_batch_val, dec_nr_val, dec_prob_val, mu_val, var_val, ambient_freq_val,reconstruction_weight=reconstruction_weight, kld_weight=kld_weight)
                        val_tot_loss += loss_minibatch.detach().item()
                        val_recon_loss += recon_loss_minibatch.detach().item()
                        val_kld_loss += kld_loss_minibatch.detach().item()

                    writer.add_scalar('ValLoss/total loss', val_tot_loss, epoch)
                    writer.add_scalar('ValLoss/reconstruction loss', val_recon_loss, epoch)
                    writer.add_scalar('ValLoss/kld_loss', val_kld_loss, epoch)
                    writer.flush()

                ################################################################################
                ## Save intermediate results every 100 epoch...
                ################################################################################
                if (epoch % plot_every_epoch == plot_every_epoch-1) and TensorBoard:

                    step = epoch // plot_every_epoch
                    with torch.no_grad():
                        z_eval, dec_nr_eval, dec_prob_eval, mu_eval, var_eval = VAE_model(self.total_set.dataset.tensors[0])

                    pr, r2, _, _ = get_correlation_btn_native_ambient(epoch, dec_prob_eval, empty_frequencies=ambient_freq_val[0,:], scRNAseq_tech=self.scRNAseq_tech)

                    writer.add_scalar('RegressionError/correlation coeffcient', pr, epoch)
                    writer.add_scalar('RegressionError/R2', r2, epoch)

                    # ..log a Matplotlib Figure showing the model's predictions on the full dataset
                    # 1, noise ratio
                    writer.add_figure('noise ratio', histgram_noise_ratio(epoch, dec_nr_eval, return_obj=True), global_step=epoch)

                    # 2, native frequencies
                    writer.add_figure('correlation', plt_correlation_btn_native_ambient(epoch, dec_prob_eval, scRNAseq_tech=self.scRNAseq_tech, empty_frequencies=ambient_freq_val[0,:], return_obj=True), global_step=epoch)
                    writer.flush()

        if save_model:
            torch.save(VAE_model, save_model)

        if TensorBoard:
            writer.add_hparams({'lr': lr, 'NN_layer1': NN_layer1, 'NN_layer2': NN_layer2, 'latent_space':latent_space, 
                           'reconstruction_weight':reconstruction_weight, 'kld_weight': kld_weight,
                          'epochs':epochs},
                          {'hparam/corr': pr, 'hparam/R2': r2})
            # writer.add_graph(model, total_set.dataset.tensors[0].cpu())
            writer.close()
        
        self.trained_model = VAE_model
        self.runtime = time.time() - training_start_time

    # Inference
    @torch.no_grad()
    def inference(self, batch_size=None):
        
        print('===========================================\n  Inferring .....')
        total_set = UMIDataset(self.raw_count, self.empty_profile)
        num_input_feature = self.num_input_feature
        sample_size = self.raw_count.shape[0]        
        self.native_counts = np.empty([sample_size, num_input_feature])
        self.bayesfactor = np.empty([sample_size, num_input_feature])
        self.native_frequencies = np.empty([sample_size, num_input_feature])
        self.noise_ratio = np.empty([sample_size, 1])
        
        if not batch_size:
            batch_size = sample_size
        i = 0
        self.total_generator = torch.utils.data.DataLoader(total_set, batch_size=batch_size, shuffle=False)
        
        for x_batch_tot, ambient_freq_tot in self.total_generator:

            minibatch_size = x_batch_tot.shape[0] # if not last batch, equals to batch size

            native_counts_batch, bayesfactor_batch, native_frequencies_batch, noise_ratio_batch = self.trained_model.inference(x_batch_tot, ambient_freq_tot[0,:])
            self.native_counts[i*batch_size:i*batch_size + minibatch_size,:] = native_counts_batch
            self.bayesfactor[i*batch_size:i*batch_size + minibatch_size,:] = bayesfactor_batch
            self.native_frequencies[i*batch_size:i*batch_size + minibatch_size,:] = native_frequencies_batch.cpu().numpy()
            self.noise_ratio[i*batch_size:i*batch_size + minibatch_size,:] = noise_ratio_batch.cpu().numpy()
            i += 1

            
class UMIDataset(torch.utils.data.Dataset):
    'Characterizes dataset for PyTorch'
    
    def __init__(self, raw_count, empty_profile, list_IDs=None):
        'Initialization'
        self.raw_count = raw_count
        self.empty_profile = empty_profile
        if list_IDs:
            self.list_IDs = list_IDs
        else:
            self.list_IDs = list(range(raw_count.shape[0]))
            
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        X1 = self.raw_count[ID,:]
        X2 = self.empty_profile[ID,:]
        return X1, X2