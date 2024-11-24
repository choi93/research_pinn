import numpy as np
import os

def createFolder(directory):
#create directory if not exist
#ex) createFolder('./test_dir')
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSERROR:
        print('Error: Creating directory. ' + directory)
  
def from_bin(fname,nx,nz):
#read binary file
#ex) seis=from_bin('seis.bin',nx,nz)
    fopen=open(fname,'rb')
    data=np.fromfile(fopen,dtype=np.float32)
    data=data.reshape(nx,nz)

    return data;

def to_bin(fname,data):
#write binary file
#ex) to_bin('result.bin',wave)
    fopen=open(fname,'wb')
    data.astype(np.float32).tofile(fopen)

