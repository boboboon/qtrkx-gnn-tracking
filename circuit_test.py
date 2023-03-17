import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq
from qcircuits.QCircuit import QCircuit
import sys,yaml
from tools.tools import *
from test import test
import cirq
from cirq.contrib.svg import SVGCircuit
import sympy

import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq
from qcircuits.QCircuit import QCircuit
import sys,yaml
from tools.tools import *
from test import test
import cirq
from cirq.contrib.svg import SVGCircuit
import sympy

sys.path.insert(0, '/Users/lucascurtin/Desktop/QGNN Repos/qtrkx-gnn-tracking')

###############################################################################
class Rescale01(tf.keras.layers.Layer):
    def __init__(self, name='Rescale01'):
        super(Rescale01, self).__init__(name=name)

    def call(self, X):
        X = tf.divide(
                tf.subtract(
                    X, 
                    tf.reduce_min(X)
                ), 
                tf.subtract(
                    tf.reduce_max(X), 
                    tf.reduce_min(X)
                ),
            lambda: X
        )
        return X
###############################################################################
class EdgeNet(tf.keras.layers.Layer):
    def __init__(self, name='EdgeNet'):
        super(EdgeNet, self).__init__(name=name)

        self.n_layers = GNN.config['EN_qc']['n_layers']
        self.n_qubits = GNN.config['EN_qc']['n_qubits']

        if 'dp_noise' in GNN.config['EN_qc'].keys():
            dp_noise = GNN.config['EN_qc']['dp_noise']
        else:
            # set noise to None if not specified 
            dp_noise = None
                
        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(IEC_id=GNN.config['EN_qc']['IEC_id'],
            PQC_id=GNN.config['EN_qc']['PQC_id'],
            MC_id=GNN.config['EN_qc']['MC_id'],
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01)
        
        self.model_circuit, self.qubits = qc.model_circuit()
        self.measurement_operators = qc.measurement_operators()

        # Prepare symbol list for inputs and parameters of the Quantum Circuits
        self.symbol_names = ['x{}'.format(i) for i in range(qc.n_inputs)]
        for i in range(qc.n_params):
            self.symbol_names.append('theta{}'.format(i)) 

        # Classical input layer of the Node Network
        # takes input data and feeds it to the PQC layer
        self.input_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_qubits, 
                activation='relu'),
            Rescale01()
        ])
        
        # Prepare PQC layer
        if (dp_noise!=None):
            # Noisy simulation requires density matrix simulator
            self.exp_layer = tfq.layers.SampledExpectation(
                cirq.DensityMatrixSimulator(noise=cirq.depolarize(dp_noise))
            )
        elif dp_noise==None and GNN.config['EN_qc']['repetitions']!=0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.SampledExpectation()
        elif dp_noise==None and GNN.config['EN_qc']['repetitions']==0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.Expectation()
        else: 
            raise ValueError('Wrong PQC Specifications!')

         # Classical readout layer
        self.readout_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        # Initialize parameters of the PQC
        self.params = tf.Variable(tf.random.uniform(
            shape=(1,qc.n_params),
            minval=0, maxval=1)*2*np.pi
        ) 

    def call(self,X, Ri, Ro):
        '''forward pass of the edge network. '''

        # Constrcu the B matrix
        bo = tf.matmul(Ro,X,transpose_a=True)
        bi = tf.matmul(Ri,X,transpose_a=True)
        # Shape of B = N_edges x 6 (2x (3 + Hidden Dimension Size))
        # each row consists of two node that are connected in the input graph.
        B  = tf.concat([bo, bi], axis=1) # n_edges x 6, 3-> r,phi,z 

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        input_to_circuit = self.input_layer(B) * np.pi

        # Combine input data with parameters in a single circuit_data matrix
        circuit_data = tf.concat(
            [
                input_to_circuit, 
                tf.repeat(self.params,repeats=input_to_circuit.shape[0],axis=0)
            ],
            axis=1
        )        
          
        # Get expectation values for all edges
        if GNN.config['EN_qc']['repetitions']==0:
            exps = self.exp_layer(
                self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data
            )
        else:
            exps = self.exp_layer(
                self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=GNN.config['EN_qc']['repetitions']
            )
    
        # Return the output of the final layer
        return self.readout_layer(exps)

class NodeNet(tf.keras.layers.Layer):
    def __init__(self, name='NodeNet'):
        super(NodeNet, self).__init__(name=name)
        
        self.n_layers = GNN.config['NN_qc']['n_layers']
        self.n_qubits = GNN.config['NN_qc']['n_qubits']

        if 'dp_noise' in GNN.config['EN_qc'].keys():
            dp_noise = GNN.config['EN_qc']['dp_noise']
        else:
            # set noise to None if not specified 
            dp_noise = None
        
        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(
            IEC_id=GNN.config['NN_qc']['IEC_id'],
            PQC_id=GNN.config['NN_qc']['PQC_id'],
            MC_id=GNN.config['NN_qc']['MC_id'],
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01
        )
        self.model_circuit, self.qubits = qc.model_circuit()
        self.measurement_operators = qc.measurement_operators()

        # Prepare symbol list for inputs and parameters of the Quantum Circuits
        self.symbol_names = ['x{}'.format(i) for i in range(qc.n_inputs)]
        for i in range(qc.n_params):
            self.symbol_names.append('theta{}'.format(i)) 

        # Classical input layer of the Node Network
        # takes input data and feeds it to the PQC layer
        self.input_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_qubits, 
                activation='relu'),
            Rescale01()
        ])

        # Prepare PQC layer
        if (dp_noise!=None):
            # Noisy simulation requires density matrix simulator
            self.exp_layer = tfq.layers.SampledExpectation(
                cirq.DensityMatrixSimulator(noise=cirq.depolarize(dp_noise))
            )
        elif dp_noise==None and  GNN.config['EN_qc']['repetitions']!=0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.SampledExpectation()
        elif dp_noise==None and  GNN.config['EN_qc']['repetitions']==0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.Expectation()
        else: 
            raise ValueError('Wrong PQC Specifications!')

        # Classical readout layer
        self.readout_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                GNN.config['hid_dim'], 
                activation='relu'),
            Rescale01()
        ])

        # Initialize parameters of the PQC
        self.params = tf.Variable(tf.random.uniform(
            shape=(1,qc.n_params),
            minval=0, maxval=1)*2*np.pi
        ) 

    def call(self, X, e, Ri, Ro):
        '''forward pass of the node network. '''

        # The following lines constructs the M matrix
        # M matrix contains weighted averages of input and output nodes
        # the weights are the edge probablities.
        bo  = tf.matmul(Ro, X, transpose_a=True)
        bi  = tf.matmul(Ri, X, transpose_a=True) 
        Rwo = Ro * e[:,0]
        Rwi = Ri * e[:,0]
        mi = tf.matmul(Rwi, bo)
        mo = tf.matmul(Rwo, bi)
        # Shape of M = N_nodes x (3x (3 + Hidden Dimension Size))
        # mi: weighted average of input nodes
        # mo: weighted average of output nodes
        M = tf.concat([mi, mo, X], axis=1)

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        input_to_circuit = self.input_layer(M) * np.pi

        # Combine input data with parameters in a single circuit_data matrix
        circuit_data = tf.concat(
            [
                input_to_circuit, 
                tf.repeat(self.params,repeats=input_to_circuit.shape[0],axis=0)
            ],
            axis=1
        )        

        # Get expectation values for all nodes
        if GNN.config['NN_qc']['repetitions']==0:
            exps = self.exp_layer(self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data)
        else:
            exps = self.exp_layer(self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=GNN.config['NN_qc']['repetitions'])

        # Return the output of the final layer
        return self.readout_layer(exps)

###############################################################################
class GNN(tf.keras.Model):
    def __init__(self):
        ''' Init function of GNN, inits all GNN blocks. '''
        super(GNN, self).__init__(name='GNN')
        # Define Initial Input Layer
        self.InputNet =  tf.keras.layers.Dense(
            GNN.config['hid_dim'], input_shape=(3,),
            activation='relu',name='InputNet'
        )
        self.EdgeNet  = EdgeNet(name='EdgeNet')
        self.NodeNet  = NodeNet(name='NodeNet')
        self.n_iters  = GNN.config['n_iters']
    
    def call(self, graph_array):
        ''' forward pass of the GNN '''
        # decompose the graph array
        X, Ri, Ro = graph_array
        # execute InputNet to produce hidden dimensions
        H = self.InputNet(X)
        # add new dimensions to original X matrix
        H = tf.concat([H,X],axis=1)
        # recurrent iteration of the network
        for i in range(self.n_iters):
            e = self.EdgeNet(H, Ri, Ro)
            H = self.NodeNet(H, e, Ri, Ro)
            # update H with the output of NodeNet
            H = tf.concat([H,X],axis=1)
        # execute EdgeNet one more time to obtain edge predictions
        e = self.EdgeNet(H, Ri, Ro)
        # return edge prediction array
        return e

def load_config(config_input,RID_input):

    
    # read the config file 
    with open(config_input, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        if len(glob.glob(config['log_dir']))==0:
            os.mkdir(config['log_dir'])
        # append RID to log dir
        config['log_dir'] = config['log_dir']+'run{}/'.format(RID_input)
        if len(glob.glob(config['log_dir']))==0:
            os.mkdir(config['log_dir'])
        # print all configs
        print('Printing configs: ')
        for key in config:
            print(key + ': ' + str(config[key]))
        print('Log dir: ' + config['log_dir'])
        print('Training data input dir: ' + config['train_dir'])
        print('Validation data input dir: ' + config['train_dir'])
        if config['run_type'] == 'new_run':
            delete_all_logs(config['log_dir'])
    # LOG the config every time
    with open(config['log_dir'] + 'config.yaml', 'w') as f:
        for key in config:
            f.write('%s : %s \n' %(key,str(config[key])))
    # return the config dictionary
    return config


import sys, os, time, datetime, csv
sys.path.insert(1, '/Users/lucascurtin/Desktop/QGNN Repos/qtrkx-gnn-tracking/tools')
from tools import *
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm
from matplotlib.lines import Line2D

import trackml.dataset

import pandas as pd

from mpl_toolkits import mplot3d


from collections import namedtuple

import numpy as np

# A Graph is a namedtuple of matrices (X, Ri, Ro, y)
Graph = namedtuple('Graph', ['X', 'Ri', 'Ro', 'y'])

def graph_to_sparse(graph):
    Ri_rows, Ri_cols = graph.Ri.nonzero()
    Ro_rows, Ro_cols = graph.Ro.nonzero()
    return dict(X=graph.X, y=graph.y,
                Ri_rows=Ri_rows, Ri_cols=Ri_cols,
                Ro_rows=Ro_rows, Ro_cols=Ro_cols)

def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.uint8):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y)

def save_graph(graph, filename):
    """Write a single graph to an NPZ file archive"""
    np.savez(filename, **graph_to_sparse(graph))

def save_graphs(graphs, filenames):
    for graph, filename in zip(graphs, filenames):
        save_graph(graph, filename)

def load_graph(filename):
    """Reade a single graph NPZ"""
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))

def load_graphs(filenames, graph_type=Graph):
    return [load_graph(f, graph_type) for f in filenames]


def sparse_to_graph(X, Ri_rows, Ri_cols, Ro_rows, Ro_cols, y, dtype=np.float32):
    n_nodes, n_edges = X.shape[0], Ri_rows.shape[0]
    Ri = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ro = np.zeros((n_nodes, n_edges), dtype=dtype)
    Ri[Ri_rows, Ri_cols] = 1
    Ro[Ro_rows, Ro_cols] = 1
    return Graph(X, Ri, Ro, y)


def map2angle(arr0):
    # Mapping the cylindrical coordinates to [0,1]
    arr = np.zeros(arr0.shape, dtype=np.float32)
    r_min     = 0.
    r_max     = 1.1
    arr[:,0] = (arr0[:,0]-r_min)/(r_max-r_min)    



   
    phi_min   = -1.0
    phi_max   = 1.0
    arr[:,1]  = (arr0[:,1]-phi_min)/(phi_max-phi_min) 
    z_min     = 0
    z_max     = 1.1
    arr[:,2]  = (np.abs(arr0[:,2])-z_min)/(z_max-z_min)  # take abs of z due to symmetry of z

    mapping_check(arr)
    return arr

n_qubits=4
circuit= cirq.Circuit()
qubits  = cirq.GridQubit.rect(n_qubits, 1)
n_layers=1
n_qubits=4
symbol_offset=0
params = sympy.symbols('theta{}:{}'.format(symbol_offset, symbol_offset + n_qubits*(1+n_layers)))

qc = QCircuit(IEC_id='simple_encoding_y',
            PQC_id='10',
            MC_id='measure_all',
            n_layers=n_layers, 
            input_size=n_qubits,
            p=0.01)


print(qc.model_circuit())