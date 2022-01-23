import argparse
import pandas as pd
import numpy as np
import contextlib
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

import os, sys
import torch
from torch.utils.tensorboard import SummaryWriter

from . import data_loader
from .vae import VAE
from .loss_functions import loss_fn
from .helper_functions import (histgram_noise_ratio,  get_correlation_btn_native_ambient,
                              plt_correlation_btn_native_ambient, assignment_accuracy,
                              naive_assignment_accuracy, plot_a_sample)

# class for writing progressbar into stdout rather than stderr
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


def run_model(train_set, val_set,
              total_set,
              num_input_feature=99,
              NN_layer1=150,
              NN_layer2=100,
              latent_space=15,
              kld_weight=1e-5,
              lr=1e-3,
              lr_step_size=5,
              lr_gamma=0.97,
              epochs = 800,
              reconstruction_weight=1,
              dropout_prob = 0,
              scRNAseq_tech='scRNAseq',
              plot_every_epoch=50,
              batch_size=64,                                                                                                
              TensorBoard=False,
              save_model=False
             ):

    '''
    raw_count_input: raw counts, numpy array
    '''
    n_batch_train = len(train_set)
    n_batch_val = len(val_set)

    # kld weight equals to 1 / n_batch_total by default
    if kld_weight is None:
        kld_weight = 1 / (n_batch_train + n_batch_val)

    # TensorBoard writer
    if TensorBoard:
        writer = SummaryWriter(TensorBoard)
        writer.add_text('Experiment description',
                        f'NN_layer1={NN_layer1}, NN_layer2={NN_layer2}, latent_space={latent_space}, kld_weight={kld_weight}, lr={lr}, epochs={epochs}, reconstruction_weight={reconstruction_weight}, dropout_prob={dropout_prob}', 0)

    # Define model
    model = VAE(num_input_feature, NN_layer1, NN_layer2, latent_space, scRNAseq_tech=scRNAseq_tech, dropout_prob=dropout_prob).cuda()

    # Define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=lr_step_size, gamma=lr_gamma)

    print("......kld_weight: ", kld_weight)
    print("......lr: ", lr)
    print("......lr_step_size: ", lr_step_size)
    print("......lr_gamma: ", lr_gamma)

    # Run training
    print('===========================================\n  Start training.....')
    

    with std_out_err_redirect_tqdm() as orig_stdout:
        # tqdm needs the original stdout
        # and dynamic_ncols=True to autodetect console width
        for epoch in tqdm(range(epochs), file=orig_stdout, dynamic_ncols=True):
    # for epoch in tqdm(range(epochs)):

            ################################################################################
            # Training
            ################################################################################
            train_tot_loss = 0
            train_kld_loss = 0
            train_recon_loss = 0

            model.train()
            for x_batch, ambient_freq in train_set:

                optim.zero_grad()
                z, dec_nr, dec_prob,  mu, var = model(x_batch)
                recon_loss_minibatch, kld_loss_minibatch, loss_minibatch = loss_fn(x_batch, dec_nr, dec_prob, mu, var, ambient_freq, reconstruction_weight=reconstruction_weight, kld_weight=kld_weight)
                loss_minibatch.backward()
                optim.step()

                train_tot_loss += loss_minibatch.detach().item()
                train_recon_loss += recon_loss_minibatch.detach().item()
                train_kld_loss += kld_loss_minibatch.detach().item()

            # ...log the running training loss
            if TensorBoard:
                writer.add_scalar('TrainingLoss/total loss', train_tot_loss, epoch)
                writer.add_scalar('TrainingLoss/reconstruction loss', train_recon_loss, epoch)
                writer.add_scalar('TrainingLoss/kld_loss', train_kld_loss, epoch)
                writer.add_scalar('learning rate', optim.param_groups[0]['lr'], epoch)
                writer.flush()

            scheduler.step()
            
            ###################################################################################
            # model evaluation
            ###################################################################################
            val_tot_loss = 0
            val_kld_loss = 0
            val_recon_loss = 0

            model.eval()
            for x_batch_val, ambient_freq_val in val_set:

                z_val, dec_nr_val, dec_prob_val,  mu_val, var_val = model(x_batch_val)
                recon_loss_minibatch, kld_loss_minibatch, loss_minibatch = loss_fn(x_batch_val, dec_nr_val, dec_prob_val, mu_val, var_val, ambient_freq_val,reconstruction_weight=reconstruction_weight, kld_weight=kld_weight)
                val_tot_loss += loss_minibatch.detach().item()
                val_recon_loss += recon_loss_minibatch.detach().item()
                val_kld_loss += kld_loss_minibatch.detach().item()

            # ...log the running validation loss
            if TensorBoard:
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
                    z_eval, dec_nr_eval, dec_prob_eval, mu_eval, var_eval = model(total_set.dataset.tensors[0])

                pr, r2, _, _ = get_correlation_btn_native_ambient(epoch, dec_prob_eval, empty_frequencies=ambient_freq_val[0,:], scRNAseq_tech=scRNAseq_tech)
                
                writer.add_scalar('RegressionError/correlation coeffcient', pr, epoch)
                # writer.add_scalar('RegressionError/MAE', mae, epoch)
                writer.add_scalar('RegressionError/R2', r2, epoch)

                # ..log a Matplotlib Figure showing the model's predictions on the full dataset
                # 1, noise ratio
                writer.add_figure('noise ratio', histgram_noise_ratio(epoch, dec_nr_eval, return_obj=True), global_step=epoch)
                #  writer.add_histogram('expected noise ratio', dec_nr_eval, step)

                # 2, native frequencies
                writer.add_figure('correlation', plt_correlation_btn_native_ambient(epoch, dec_prob_eval, scRNAseq_tech=scRNAseq_tech, empty_frequencies=ambient_freq_val[0,:], return_obj=True), global_step=epoch)
                writer.flush()
                    
    if save_model:
        torch.save(model, save_model)
        
    if TensorBoard:
        writer.add_hparams({'lr': lr, 'NN_layer1': NN_layer1, 'NN_layer2': NN_layer2, 'latent_space':latent_space, 
                       'reconstruction_weight':reconstruction_weight, 'kld_weight': kld_weight,
                      'epochs':epochs},
                      {'hparam/corr': pr, 'hparam/R2': r2})
        # writer.add_graph(model, total_set.dataset.tensors[0].cpu())
        writer.close()
    
    # Inference
    print('===========================================\n  Inferring .....')
    
    with torch.no_grad():
        
        sample_size = total_set.dataset.tensors[0].shape[0]
        expected_native_counts = np.empty([sample_size, num_input_feature])
        probs = np.empty([sample_size, num_input_feature])
        expected_native_frequencies = np.empty([sample_size, num_input_feature])
        expected_noise_ratio = np.empty([sample_size, 1])
        i = 0
        
        for x_batch_tot, ambient_freq_tot in total_set:
            
            minibatch_size = x_batch_tot.shape[0] # if not last batch, equals to batch size
            
            expected_native_counts_batch, probs_batch, expected_native_frequencies_batch, expected_noise_ratio_batch = model.inference(x_batch_tot, ambient_freq_tot[0,:])
            expected_native_counts[i*batch_size:i*batch_size + minibatch_size,:] = expected_native_counts_batch
            probs[i*batch_size:i*batch_size + minibatch_size,:] = probs_batch
            expected_native_frequencies[i*batch_size:i*batch_size + minibatch_size,:] = expected_native_frequencies_batch.cpu().numpy()
            expected_noise_ratio[i*batch_size:i*batch_size + minibatch_size,:] = expected_noise_ratio_batch.cpu().numpy()
            
            i += 1
    
    return expected_native_counts, probs, expected_native_frequencies, expected_noise_ratio


def main():
    
    ##########################################################################################
    ## argument parser
    ##########################################################################################

    parser = argparse.ArgumentParser(description='single cell Ambient Remover (scAR): remove ambient signals for scRNAseq data')
    parser.add_argument('count_matrix', type=str, nargs='+', help='the file of observed count matrix, 2D array (cells x genes)')
    parser.add_argument('-e','--empty_profile', type=str, default=None, help='the file of empty profile obtained from empty droplets, 1D array')
    parser.add_argument('-t', '--technology', type=str, default='scRNAseq', help='scRNAseq technology, e.g. scRNAseq, CROPseq, CITEseq, ... etc.')
    parser.add_argument('-o', '--output', type=str, default=None, help='output directory')
    parser.add_argument('-tb', '--TensorBoard', type=str, default=False, help='Tensorboard directory')
    
    parser.add_argument('-hl1', '--hidden_layer1', type=int, default=None, help='number of neurons in the first layer')
    parser.add_argument('-hl2', '--hidden_layer2', type=int, default=None, help='number of neurons in the second layer')
    parser.add_argument('-ls', '--latent_space', type=int, default=None, help='dimension of latent space')
    parser.add_argument('-epo', '--epochs', type=int, default=800, help='training epochs')
    parser.add_argument('-s', '--save_model', type=int, default=False, help='whether save the trained model')
    parser.add_argument('-plot', '--plot_every_epoch', type=int, default=50, help='plot every epochs')
    parser.add_argument('-batchsize', '--batchsize', type=int, default=64, help='batch size')

    
    args = parser.parse_args()

    count_matrix_path = args.count_matrix[0]
    empty_profile_path = args.empty_profile
    scRNAseq_tech = args.technology
    output_dir = os.getcwd() if not args.output else args.output  # if None, output to current directory
    TensorBoard = args.TensorBoard
    NN_layer1 = args.hidden_layer1
    NN_layer2 = args.hidden_layer2
    latent_space = args.latent_space
    epochs = args.epochs
    save_model = args.save_model
    plot_every_epoch = args.plot_every_epoch
    batch_size = args.batchsize
    
    print('===========================================')
    print('scRNAseq_tech: ', scRNAseq_tech)
    print('output_dir: ', output_dir)
    print('count_matrix_path: ', count_matrix_path)
    print('empty_profile_path: ', empty_profile_path)
    print('TensorBoard path: ', TensorBoard)
    
    # Read data
    print('===========================================\n  Reading data...')
    print('-------------------------------------------')
    count_matrix = pd.read_pickle(count_matrix_path)
    count_matrix = count_matrix.fillna(0) # replace missing values with zeros
    print('  ... count_matrix:')
    count_matrix.info(max_cols=10)
    
    if args.empty_profile:
        empty_profile = pd.read_pickle(empty_profile_path)
        print(' ... calculate empty profile using empty droplets')
        assert (empty_profile.index == count_matrix.columns).all()        
    else:
        empty_profile = count_matrix.sum(axis=0)/count_matrix.sum().sum()
        empty_profile = empty_profile.to_frame()
        print(' ... calculate empty profile using cell-containing droplets')
        
    print('-------------------------------------------')
    print(' ... empty_profile:')
    empty_profile = empty_profile.fillna(0) # replace missing values with zeros
    empty_profile.info(max_cols=10)
    
    print('===========================================\n  Loading data to dataloader...')
    train_set, val_set, total_set = data_loader.get_dataset(count_matrix.values, empty_profile.values, split=0.002, batch_size=batch_size)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Run model
    expected_native_counts, probs, expected_native_frequencies, expected_noise_ratio = run_model(train_set,
                                                                                                 val_set,
                                                                                                 total_set,
                                                                                                 num_input_feature=count_matrix.shape[1],
                                                                                                 NN_layer1=NN_layer1,
                                                                                                 NN_layer2=NN_layer2,
                                                                                                 latent_space=latent_space,
                                                                                                 scRNAseq_tech=scRNAseq_tech,
                                                                                                 epochs=epochs,
                                                                                                 plot_every_epoch=plot_every_epoch,
                                                                                                 batch_size=batch_size,
                                                                                                 TensorBoard=TensorBoard,
                                                                                                 save_model=save_model
                                                                                                )

    print('===========================================\n  Saving results...')
    output_path01, output_path02, output_path03, output_path04 = os.path.join(output_dir, f'expected_natives.pickle'), os.path.join(output_dir, f'probs.pickle'), os.path.join(output_dir, f'expected_native_freq.pickle'), os.path.join(output_dir, f'expected_noise_ratio.pickle')
  
    # save results
    pd.DataFrame(expected_native_counts, index=count_matrix.index, columns=count_matrix.columns).to_pickle(output_path01)
    pd.DataFrame(probs, index=count_matrix.index, columns=count_matrix.columns).to_pickle(output_path02)
    pd.DataFrame(expected_native_frequencies, index=count_matrix.index, columns=count_matrix.columns).to_pickle(output_path03)
    pd.DataFrame(expected_noise_ratio, index=count_matrix.index, columns=['expected_noise_ratio']).to_pickle(output_path04)
    
    print(f'...denoised counts saved in: {output_path01}')
    print(f'...probability matrix saved in: {output_path02}')
    print(f'...expected native frequencies saved in: {output_path03}')
    print(f'...expected noise ratio saved in: {output_path04}')
    
    print(f'===========================================\n  Done!!!')

if __name__ == "__main__":
    main()