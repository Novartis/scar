import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

# helper function to prepare datasets for training
def get_dataset(input_raw_counts, expected_ambient_frequency, split=0.2, batch_size=64):
    '''
    Load obs_count data and empty profile, return two datasets: train_set and val_set    
    '''
    # get sample size
    sample_size = input_raw_counts.shape[0]
    
    # expand ambient frequency to the same dimentions as input raw counts.
    if expected_ambient_frequency.squeeze().ndim == 1:
        expected_ambient_frequency = expected_ambient_frequency.squeeze().reshape(1,-1).repeat(sample_size, axis=0)
    
    # ambient_frequeny data in tensor
    ambient_freq_tensor = torch.tensor(expected_ambient_frequency, dtype=torch.float32).cuda()
    
    # input_raw_counts= np.log(input_raw_counts/np.median(input_raw_counts.sum(axis=1))+1)  # test
    # input_raw_counts = (input_raw_counts - np.mean(input_raw_counts))/np.std(input_raw_counts) # test
    # count data in tensor
    raw_count_tensor = torch.tensor(input_raw_counts, dtype=torch.float32).cuda()
    
    # create the dataset
    dataset = TensorDataset(raw_count_tensor, ambient_freq_tensor)

    # determine training and validation size
    split = split
    val_sample_size = int(split*sample_size)
    train_sample_size = sample_size - val_sample_size
    
    # split data into training and validation sets
    train_set, val_set = random_split(dataset, (train_sample_size, val_sample_size))
    
    # load datasets to dataloader
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    total_set = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return train_set, val_set, total_set
