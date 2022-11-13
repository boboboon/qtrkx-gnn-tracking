from numpy import load
import sys
import os
# Turn off warnings and errors due to TF libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import time
import datetime
import csv
from random import shuffle
import tensorflow as tf
# import internal scripts
from tools.tools import *
from test import test


def test():
    # This load_config function comes from our tools script, lets us load up our arguments which we'll use later.
    # because config file is in yaml format.

    config = load_config(parse_args())
    tools.config = config


    # This section is just for GPU stuff I assume for processing, not taking a look at this just yet.

    # Set GPU variables
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    USE_GPU = (config['gpu']  != '-1')

    # Set number of thread to be used
    os.environ['OMP_NUM_THREADS'] = str(config['n_thread'])  # set num workers
    tf.config.threading.set_intra_op_parallelism_threads(config['n_thread'])
    tf.config.threading.set_inter_op_parallelism_threads(config['n_thread'])


    # Next we check in our config file for this training which network we're using.
    # Let's say we're using the CGNN, we load up the specs from the qnetworks file
    # This includes the input net, edge net and node net layers and they're sequentially put together with TF
    if config['network'] == 'QGNN':
        from qnetworks.QGNN import GNN
        GNN.config = config
    elif config['network'] == 'CGNN':
        from qnetworks.CGNN import GNN
        GNN.config = config
    else: 
        print('Wrong network specification!')
        sys.exit()

    # setup model
    model = GNN()



    # We're next going to use the get_dataset function which is again within the tools but we're defining below.
    # The input directory is where we're getting the training data from i.e. train/data
    # The n_files is specified in the config file and is our number of files


    def get_dataset(input_dir,n_files):
        return GraphDataset(input_dir, n_files)


    # This GraphDataset is a python class defined in the tools section and can be found below
    # When the train_data is defined 

    class GraphDataset():
        def __init__(self, input_dir, n_samples=None):
            input_dir = os.path.expandvars(input_dir)
            filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                        if f.startswith('event') and f.endswith('.npz')]
            self.filenames = (
                filenames[:n_samples] if n_samples is not None else filenames)

        def __getitem__(self, index):
            return load_graph(self.filenames[index])

        def __len__(self):
            return len(self.filenames)


    # When the function is used in train.py, they're using n_train as 50 despite there being 51 training files (there are 51 validation
    # files also). This means the files have been roughly evenly split up into training and validation sets but it's odd they don't
    # add to 100? They don't seem to call n_files anywhere in the training script

    train_data = get_dataset(config['train_dir'], config['n_train'])
    train_list = [i for i in range(config['n_train'])]


