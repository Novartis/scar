# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, median_absolute_error
from scipy.stats import pearsonr

## plot estimated noise ratio
def histgram_noise_ratio(epoch, nr_pred, nr_truth=None, xlim=(0.5, 1), return_obj=False): 
    
    epsilon = nr_pred.detach().cpu().numpy().squeeze()
    nosie_df = pd.DataFrame(epsilon, columns=['noise ratio'])
    
    fig = plt.figure(figsize=(3,2), tight_layout=True);
    ax = sns.histplot(nosie_df['noise ratio'], bins=200, label='inferenced');

    plt.legend();
    plt.title(f'epoch: {epoch}')
    # plt.xlim(xlim)
    plt.xlabel('noise ratio')
    plt.ylabel('cell counts')

    if nr_truth is not None:
        nosie_df["ground truth"] = nr_truth
        ax.vlines(nr_truth, ymin=0., ymax=180, colors='r', ls='--', lw=2, label='ground truth');
        
    if return_obj:
        return fig

def jointplot_noise_ratio(epoch, nr_pred, nr_truth, celltype, xlim=(0.3, 0.55), return_obj=False): 

    noise_ratio_inf = nr_pred.cpu().numpy().squeeze()
    nosie_df = pd.DataFrame(np.array([noise_ratio_inf, nr_truth]).T, columns=['inferenced', 'ground truth'])
    nosie_df["celltype"] = celltype

    ax = sns.jointplot(data=nosie_df, x="ground truth", y="inferenced", hue="celltype", kind='scatter', s=2, alpha=0.5);
    ax.ax_joint.plot(xlim, xlim, 'b--', linewidth = 2);
    plt.tight_layout()

    if return_obj:
        return ax.fig
    
def get_correlation_btn_native_ambient(epoch, native_signals_pred, empty_frequencies, scRNAseq_tech):
    
    # assign the highest guide
    def maxvalue(x):
        return (x == x.max(dim=1).values.view(-1,1)).type(x.dtype)
    
    if scRNAseq_tech.lower() == 'cropseq':
        native_clean = maxvalue(native_signals_pred.detach()).cpu().numpy()
    else:
        native_clean = native_signals_pred.cpu().numpy()
        
    frac_native = native_clean.sum(axis=0)/native_clean.sum()
    freq_truth = empty_frequencies.detach().cpu().numpy()
    
    # metric to evaluate the predection
    r2 = r2_score(freq_truth, frac_native)
    # mae = median_absolute_error(freq_truth, frac_native)
    pr = pearsonr(freq_truth, frac_native)[0]

    return pr, r2, frac_native, freq_truth
    
    
def plt_correlation_btn_native_ambient(epoch, native_frequencies_pred, empty_frequencies, scRNAseq_tech, return_obj=False):

    pr, r2, frac_native, freq_truth = get_correlation_btn_native_ambient(epoch, native_frequencies_pred, empty_frequencies, scRNAseq_tech=scRNAseq_tech)
    
    fig = plt.figure(figsize=(3,3), tight_layout=True);
    plt.title(f'epoch: {epoch}')
    plt.scatter(freq_truth, frac_native, s=10);
    xmin, xmax, ymin, ymax = plt.axis()
    plt.plot([xmin, xmax], [ymin, ymax], 'r--');
    plt.text(1.15*xmin, 0.85*ymax, f'corr: {pr:.4f} \nR2: {r2:.4f}\n')
    plt.xlabel('empty profile');
    plt.ylabel('expected native frequencies');
    
    if return_obj:
        return fig

def assignment_accuracy(probs, celltype_truth):
    return (np.argmax(probs, axis=1) == celltype_truth).sum()/len(celltype_truth)

def naive_assignment_accuracy(count_matrix, celltype_truth):
    row_ct = np.argmax(count_matrix, axis=1)
    return (row_ct == celltype_truth).sum()/len(celltype_truth)

def plot_a_sample(epoch, native_frequencies_pred, syn_data_obj, idx=None, return_obj=False):
    
    if idx is None:
        idx = np.random.randint(syn_data_obj.n_samples)
    
    plt.figure(figsize=(8,12))
    plt.subplot(5,1,1)
    plt.title(f'cell: {idx}   epoch: {epoch}')
    plt.scatter(np.linspace(0, syn_data_obj.n_fbs-1, syn_data_obj.n_fbs), syn_data_obj.ambient_signals[idx], c='gray', alpha=0.5, label='ambient signals');
    plt.scatter(syn_data_obj.celltype[idx], syn_data_obj.count_matrix[idx][syn_data_obj.celltype[idx]], c='b', alpha=0.5, label='observation');
    plt.scatter(syn_data_obj.celltype[idx], syn_data_obj.native_signals[idx][syn_data_obj.celltype[idx]], c='r', alpha=0.5, label='native signals');
    plt.ylabel('counts');
    plt.legend();

    plt.subplot(5,1,2)
    plt.scatter(np.linspace(0, n_fbs-1, n_fbs), syn_data_obj.empty_profile, c='gray', alpha=0.5, label='ambient frequencies');
    plt.scatter(np.linspace(0, n_fbs-1, n_fbs), native_frequencies_pred.cpu().numpy()[idx], c='r', alpha=0.5, label='native frequencies');
    plt.ylabel('frequencies')
    plt.legend()
    
    if return_obj:
        return plt.gcf()
    
def barplot_native_freq_by_celltype(epoch, dec_prob, syn_data_obj, return_obj=False):
    n_fbs = syn_data_obj.n_fbs
    n_celltypes = syn_data_obj.n_celltypes
    fig, axs = plt.subplots(ncols=n_celltypes, figsize=(n_celltypes*5, 5), sharey=True, tight_layout=True);
    type_specific_profile = dec_prob.cpu()

    for i in np.unique(syn_data_obj.celltype):
        axs[i].barh(y = np.linspace(1,n_fbs,n_fbs), width = syn_data_obj.native_profile[syn_data_obj.celltype==i][0,...], height=0.9, alpha=0.2);
        axs[i].boxplot(type_specific_profile[syn_data_obj.celltype==i].T, vert=False, notch=False, widths=0.7, medianprops=dict(linestyle='-.', linewidth=2.5, color='firebrick'));
        axs[i].set_ylabel("ADT or genes")
        axs[i].set_title(f"cell type {i}")
        axs[i].text(0.6*axs[i].get_xlim()[1], 1, f"box: inference \nbar: ground truth")
        axs[i].set_xlabel("frequencies")
    if return_obj:
        return fig

def heatmap_native_pred(expected_native_counts, syn_data_obj, return_obj=False):

    idx_ct_sorted = np.argsort(syn_data_obj.celltype)

    fake_signals_nv = syn_data_obj.native_signals
    fake_signals_bg = syn_data_obj.ambient_signals
    fake_signals_Obs = syn_data_obj.count_matrix

    log_native_signals = np.log2(fake_signals_nv + 1)
    log_ambient_sigals = np.log2(fake_signals_bg + 1)
    log_obs = np.log2(fake_signals_Obs + 1)
    
    fig, axs = plt.subplots(ncols=4, nrows=1, figsize=(12,3),tight_layout=True);

    sns.heatmap(log_obs[idx_ct_sorted],yticklabels=False,
                     vmin=0, vmax=10, cmap="coolwarm", center=2,ax=axs[0]);
    axs[0].set_title("Observation");
    axs[0].set_ylabel("cells");
    axs[0].set_xlabel("protein markers");
    sns.heatmap(log_ambient_sigals[idx_ct_sorted], yticklabels=False,
                     vmin=0, vmax=10, cmap="coolwarm", center=2,ax=axs[1]);
    axs[1].set_title("ambient signals");
    axs[1].set_xlabel("protein markers");

    sns.heatmap(log_native_signals[idx_ct_sorted], yticklabels=False,
                     vmin=0, vmax=10, cmap="coolwarm", center=2,ax=axs[2]);
    axs[2].set_title("native signals");
    axs[2].set_xlabel("protein markers");

    sns.heatmap(np.log2(expected_native_counts.cpu().numpy()+1)[idx_ct_sorted], yticklabels=False,
                     vmin=0, vmax=10, cmap="coolwarm", center=2,ax=axs[3]);
    axs[3].set_title("Expected native signals");
    axs[3].set_xlabel("protein markers");
    
    if return_obj:
        return fig