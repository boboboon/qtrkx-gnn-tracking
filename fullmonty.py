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
from train_script import main






# So we want a script that trains a CGNN 5 times (updating the RID each time), then changes the number of iterations (increase from 2, 3 times i.e.: 1,3,5,7)

RID_input=['1','2','3','4','5']


config_input_list=[r'configs/CGNN_iter1.yaml',r'configs/CGNN_iter3.yaml',r'configs/CGNN_iter5.yaml',r'configs/CGNN_iter7.yaml']

for i in range(1):
    config=config_input_list[i]
    iter=2*i+1
    for v in range(1):
        RID='{}-{}'.format(iter,v)
        print(config)
        print(RID)
        main(config,RID)


   





#for i in whatever:
    #main(config[i], whatever)
