import SimpleITK as sitk
import numpy as np
import sys

import dicom_lister as gdcm_lister2
import pydicom as dicom
from tensorflow.keras.models import load_model
import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.ndimage import gaussian_filter
gaussian_sigma=1.5

d='input'
im_dir='prediction_screens'

def load_ct_for_weight(ct):
    xextent,yextent,zextent=500.,500.,1500.
    xdim,ydim,zdim=128,128,256
    resample_extent=(xextent,yextent,zextent)
    zeds=np.zeros((zdim,ydim,xdim))
    final_resample_spacing=np.array((xextent/xdim,yextent/ydim,zextent/zdim))
    rs=sitk.ResampleImageFilter()
    origin=np.array(ct.GetOrigin())
    original_dims=np.array(ct.GetSize())
    original_spacing=np.array(ct.GetSpacing())
    original_extent=original_dims*original_spacing
    origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
    origin[2]=origin[2]-origin_shift
    delta_extent=resample_extent-original_extent
    delta_x=delta_extent[0]/2.
    delta_y=delta_extent[1]/2.
    new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
    ref=sitk.GetImageFromArray(zeds)
    ref.SetSpacing(final_resample_spacing)
    ref.SetOrigin(new_origin)
    rs.SetReferenceImage(ref)
    rs.SetDefaultPixelValue(-1000)
    rs.SetInterpolator(sitk.sitkLinear)
    ct_midres = rs.Execute(ct)
    x = sitk.GetArrayFromImage(ct_midres)
    x[x<-1024]=-1024 #fix for very low values
    out=np.zeros((zdim,xdim*2))
    out[:,0:xdim]=np.average(x,axis=1)
    out[:,xdim:]=np.average(x,axis=2)
    out=np.expand_dims(np.expand_dims(out,0),-1)
    #print('Flattened for Weight prediction...')
    return out        

model_path="models/256_500_500_1500-128_128_256_EfficientNetB2_False_mse_64.hdf5"
model=load_model(model_path,compile=False)

df=gdcm_lister2.nested_dir_to_df(d)
uids=np.unique(df.FrameOfReferenceUID.values)
reader=sitk.ImageSeriesReader()
dfo=pd.DataFrame(columns=['Patient','Date','SeriesDescription','PredictedWeight','SeriesUID'])
i=0

import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
cmap = pl.cm.gnuplot2
my_cmap = cmap(np.arange(cmap.N))
my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
my_cmap = ListedColormap(my_cmap)

for uid in uids:
    ct_fnames=df[(df.FrameOfReferenceUID==uid) & (df.Modality=='CT')].Filenames.values[0]
##    series_uid=df[(df.FrameOfReferenceUID==uid) & (df.Modality=='CT')].SeriesUID.values[0]
##    pt_fnames=df[(df.FrameOfReferenceUID==uid) & (df.Modality=='PT')].Filenames.values[0]    
    reader.SetFileNames(ct_fnames)
    ct=reader.Execute()
    dcm=dicom.read_file(ct_fnames[0])
    pat=str(dcm.PatientName)
    try:
        desc=str(dcm.SeriesDescription)
    except:
        desc=''
    try:
        date=str(dcm.AcquisitionDate)
    except:
        date=''
##    weight=float(dcm.PatientWeight)
    x=load_ct_for_weight(ct)
    pred=model.predict(x)[0][0]
    row=[pat,date,desc,pred,uid]
    dfo.loc[i]=row
    print(i,pat,date,desc,pred,uid)

    x=tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        outputs=model(x,training=False)
    grads=tape.gradient(outputs,x)
    dgrad_abs = tf.math.abs(grads)
    g=gaussian_filter(dgrad_abs[0,...,0],gaussian_sigma)

    arr_min, arr_max  = np.min(g), np.max(g)
    grad_eval = (g - arr_min) / (arr_max - arr_min + 1e-18)

    
    plt.figure(figsize=[12,12])
    plt.imshow(np.flipud(x[0,...,0]),cmap='Greys_r',aspect=1.5)
    plt.axis('off')
    #plt.imshow(np.flipud(g),cmap='gnuplot2',alpha=0.5,aspect=1.5)
    #plt.imshow(np.flipud(g),cmap='gnuplot2',alpha=0.3,aspect=1.5,clim=[0.,0.0005])
    #plt.imshow(np.flipud(grad_eval),cmap='gnuplot2',alpha=0.5,aspect=1.5)
    plt.imshow(np.flipud(grad_eval),cmap=my_cmap,alpha=0.75,aspect=1.5)
    title=pat+' Predicted Weight (kg): '+str(round(pred,1))
    plt.title(title,fontsize=20)
    plt.tight_layout()
    plt.savefig(join(im_dir,str(uid).zfill(4)+'.jpg'))
    plt.close('all')
    i+=1

dfo.to_csv('predicted_weights.csv')
