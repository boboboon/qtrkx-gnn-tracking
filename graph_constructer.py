import sys, os, time, datetime, csv

from tools.tools import *
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


#Get our filenames
file_path=r'/Users/lucascurtin/Desktop/edge_data/preprocessed_edges_1000_1099'

filenames=os.listdir(file_path)

for i in range(len(filenames)):
    #Get our df for the filename

    file_name=filenames[i]

    edge_df_path=file_path+'/{}'.format(file_name)

    cols = [ 'edge', 'label', 'eta', 'phi', 'true_pt', 'layer', 'track_length', 'hit1_id', 'hit2_id', 'particle_num'] #suggested column names

    event = pd.DataFrame(np.load(edge_df_path,allow_pickle = True), columns = cols)

    true_events=event.loc[event['label'] == 1]

    fake_events=event.loc[event['label'] == 0]


    #Get our total input and output coordinates

    input_coords=[]
    output_coords=[]

    for i in range(len(event)):
        input_i=event['edge'].iloc[i][0:3]
        input_coords.append(input_i)

    input_coords=np.array(input_coords)

    for i in range(len(event)):
        output_i=event['edge'].iloc[i][3:6]
        output_coords.append(output_i)

    output_coords=np.array(output_coords)


    #Combine our input coordinates and find all unique values
    IO = np.concatenate((input_coords,output_coords),axis=0)

    # create a set of tuples representing the rows of the matrix
    unique_rows = set(tuple(row) for row in IO)

    # convert the set of tuples back to a list of lists
    unique_matrix = [list(row) for row in unique_rows]


    unique_matrix = []

    # iterate over each unique row and append it to the new matrix
    for row in unique_rows:
        unique_matrix.append(list(row))



    # Start creating our R matrices:

    X_new=unique_matrix

    #So now we go through each row of our X matrix, grab the coordinates and see where they crop up in the input part of the edge coordinates. 
    n_hits=len(X_new)
    n_edges=len(event)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)


    #So now we go through each row of our X matrix, grab the coordinates and see where they crop up in the input part of the edge coordinates. 
    n_hits=len(X_new)
    n_edges=len(event)
    Ri = np.zeros((n_hits, n_edges), dtype=np.uint8)
    Ro = np.zeros((n_hits, n_edges), dtype=np.uint8)

    for i in range(len(X_new)):

        #Load our unique hit
        input=X_new[i]

        #Find out where our hit is in our edge dataframe (shows up in the first 3, we've done this with the input_coords cell)
        index_R=[]
        index = np.where(input_coords == input)
        unique_index=np.unique(index[0])
        index_list=index[0].tolist()

        for v in range(len(unique_index)):
            count_n=index_list.count(unique_index[v])
            if count_n==3:
                index_R.append(unique_index[v])

        #Now we need to set the row of Ri corresponding to this hit with the corresponding input information
        row=Ri[i]
        column_vals=index_R

        for c in range(len(column_vals)):
            Ri[i,column_vals[c]]=1



    #Ro now

    for i in range(len(X_new)):

        #Load our unique hit
        input=X_new[i]

        #Find out where our hit is in our edge dataframe (shows up in the first 3, we've done this with the input_coords cell)
        index_R=[]
        index = np.where(output_coords == input)
        unique_index=np.unique(index[0])
        index_list=index[0].tolist()

        for v in range(len(unique_index)):
            count_n=index_list.count(unique_index[v])
            if count_n==3:
                index_R.append(unique_index[v])

        #Now we need to set the row of Ri corresponding to this hit with the corresponding input information
        row=Ro[i]
        column_vals=index_R

        for c in range(len(column_vals)):
            Ro[i,column_vals[c]]=1


    #Finally, each column in these matrices represents an edge, so the y values we can pluck directly from the edge df

    y=np.array(event['label'])
    y



    X_input=np.array(X_new)
    Ri_input=Ri.astype('float32')
    Ro_input=Ro.astype('float32')
    y_input=y.astype('float32')
    X_input[:,0]=X_input[:,0]/1000
    X_input[:,1]=X_input[:,1]/np.pi
    X_input[:,2]=X_input[:,2]/1000
    X_input[:,2]


    save_path=r'/Users/lucascurtin/Desktop/edge_data/graphs_marcin/'

    hit_name=file_name[-13:-4]
    file_save='event{}_g000.npz'.format(hit_name)


    graph_i=Graph(X_input,Ri_input,Ro_input,y_input)
    save_graph(graph_i,'/Users/lucascurtin/Desktop/edge_data/train_graphs/{}'.format(file_save))
    print('Saved file:',file_save)






