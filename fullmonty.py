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
import train_script



cwd=os.getcwd()
print("HERE:",cwd)


train_script.main('configs/test_CGNN.yaml','5')


#for i in range(1):
    #config=config_input_list[i]
    #iter=2*i+1
    #for v in range(1):
       # RID='{}-{}'.format(iter,v)
       # print(config)
       # print(RID)
       # main(config,RID)


   





#for i in whatever:
    #main(config[i], whatever)
