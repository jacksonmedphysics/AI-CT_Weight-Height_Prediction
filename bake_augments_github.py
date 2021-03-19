import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from os.path import join
import SimpleITK as sitk
from scipy import ndimage
#from numpy.random import rand
import random
from random import random as rand #overwrite np random
import datetime
from multiprocessing import Pool

#loading the 3D scans, augmenting, and flattening is computationally slow. If training multiple models, it's sensible to bake the augmentations and have many
#representations of the 2D (small file) trianing data ready to load when optimising. 

"""
python bake_augments_github.py
"""

out_dir='baked_training_256' #output folder for baked validation and augmented data
ct_dir='ct' #folder containing 3D CT images as .nii.gz format, eg Case001.nii.gz 
df=pd.read_csv('sample_weight_spreadsheet.csv',index_col=0) #cases to be used should appear in this csv (also used for training with body weight), will generate list of cases to loop through
uids=df.SeriesUID.values 
weights=df.PatientWeight.values

n_processes=16
batch_size=16
n_augments_this_pass=20 #how many augments to run on this script call. Can re-run and will start from highest aug # detected so won't overwrite
#if set to 20, can run this script 3x times and have 60 versions of each case. So easy to add more augmented data if it seems necessary for training
bake_validation=True #only needs to be flagged on first run, creates the [CASE]---val.nii.gz version of each training case

xextent=float(500) #physical extents in mm, pads out (or potentially crops slightly) all CTs into a consistent "phone-booth"
yextent=float(500)
zextent=float(1500)
axial_dim=128 #axial (x/y) resolution
xdim=int(axial_dim)
ydim=int(axial_dim)
zdim=int(256) #z-axis resolution to use, output file will be (2*axial_dim,zdim) resolution, eg 256x256

zeds=np.zeros((zdim,ydim,xdim))
xy_max_crop_dist=70 #mm
resample_extent=(xextent,yextent,zextent)  #max extent was 1942
final_resample_dims=(zdim,ydim,xdim)  #numpy zyx
final_resample_spacing=np.array((resample_extent[0]/final_resample_dims[2],resample_extent[1]/final_resample_dims[1],resample_extent[2]/final_resample_dims[0])) #sitk xyz
rss=final_resample_spacing


def random_crop(image):
    random_crop_pct=0.5
    if True:
    #try:
        #ar=sitk.GetArrayFromImage(image)
        spacing=image.GetSpacing()
        cropfilter=sitk.CropImageFilter()
        x_crop=int(xy_max_crop_dist/spacing[0]*rand())
        y_crop=int(xy_max_crop_dist/spacing[1]*rand())
        #midz=ar.shape[0]*np.random.rand()#ar.shape[0]/2
        midz = image.GetSize()[2] * rand()  # ar.shape[0]/2
        z_lower_crop=int(random_crop_pct*midz*rand())
        #z_upper_crop=int(random_crop_pct*(ar.shape[0]-midz)*rand())
        z_upper_crop = int(random_crop_pct * (image.GetSize()[2] - midz) * rand())
        cropfilter.SetLowerBoundaryCropSize([x_crop,y_crop, z_lower_crop])
        cropfilter.SetUpperBoundaryCropSize([x_crop,y_crop, z_upper_crop])
        image_crop = cropfilter.Execute(image)
        return image_crop

def augment_case(x,reference_im):
    ct=x
    max_rotation_deg=5 #20.
    max_translation=7.  #25
    max_gauss_sigma=1.0
    max_hu_shift=30
    max_noise=100
    sharpening_range=0.6
    sharpening_alpha=0.5
    rotx=(rand()-0.5)*max_rotation_deg*2*3.1415/360
    roty=(rand()-0.5)*max_rotation_deg*2*3.1415/360
    rotz=(rand()-0.5)*max_rotation_deg*2*3.1415/360
    tx=(rand()-0.5)*max_translation
    ty=(rand()-0.5)*max_translation
    tz=(rand()-0.5)*max_translation
    sig=rand()*max_gauss_sigma
    sharp=sharpening_range*(1-0.3*(rand()-0.5))
    salpha=sharpening_alpha*(1-0.8*(rand()-0.5))
    hu_shift=(rand()-0.5)*max_hu_shift
    #img=sitk.GetImageFromArray(ct)
    #img.SetSpacing(final_resample_spacing)
    img=x
    initial_transform=sitk.Euler3DTransform()
    initial_transform.SetParameters((rotx,roty,rotz,tx,ty,tz))
    img = sitk.Resample(img, reference_im, initial_transform, sitk.sitkLinear, -1000., sitk.sitkInt16)
    ar=sitk.GetArrayFromImage(img)
    blurred_ar=ndimage.gaussian_filter(ar,sharp)
    sharpened=ar+salpha*(ar-blurred_ar)
    ar=sharpened
    ar=ndimage.gaussian_filter(ar,sigma=sig)
    ar+=int(hu_shift)
    np.random.seed(random.randint(0,65535))
    ar+=((np.random.random(ar.shape)-0.5)*max_noise).astype('int16')
    return ar.astype('float32')

def load_case(label_name):
    while(True):
        try:
            ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name+'.nii.gz')),sitk.sitkInt16)
            rs=sitk.ResampleImageFilter()
            if True:
                ct=random_crop(ct) #applies random cropping to CT/Label combo
            origin=np.array(ct.GetOrigin())
            original_dims=np.array(ct.GetSize())
            original_spacing=np.array(ct.GetSpacing())
            original_extent=original_dims*original_spacing
            if True:
                origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
            origin[2]=origin[2]-origin_shift
            delta_extent=resample_extent-original_extent
            delta_x=delta_extent[0]/2.
            delta_y=delta_extent[1]/2.
            new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
            ref=sitk.GetImageFromArray(zeds)
            ref.SetSpacing(final_resample_spacing)
            ref.SetOrigin(new_origin)
            if True:
                x=augment_case(ct,ref)
            else:
                rs.SetReferenceImage(ref)
                rs.SetDefaultPixelValue(-1000)
                rs.SetInterpolator(sitk.sitkLinear)
                ct_midres = rs.Execute(ct)
                x = sitk.GetArrayFromImage(ct_midres)
            x[x<-1024]=-1024 #fix for very low values
            out=np.zeros((zdim,xdim*2))
            out[:,0:xdim]=np.average(x,axis=1)
            out[:,xdim:]=np.average(x,axis=2)
            return out
        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            break

def load_val_case(label_name):
    while(True):
        try:
            ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name+'.nii.gz')),sitk.sitkInt16)
            rs=sitk.ResampleImageFilter()
            # if True:
            #     ct=random_crop(ct) #applies random cropping to CT/Label combo
            origin=np.array(ct.GetOrigin())
            original_dims=np.array(ct.GetSize())
            original_spacing=np.array(ct.GetSpacing())
            original_extent=original_dims*original_spacing
            if True:
                origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
            origin[2]=origin[2]-origin_shift
            delta_extent=resample_extent-original_extent
            delta_x=delta_extent[0]/2.
            delta_y=delta_extent[1]/2.
            new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
            ref=sitk.GetImageFromArray(zeds)
            ref.SetSpacing(final_resample_spacing)
            ref.SetOrigin(new_origin)
            # if False:
            #     x=augment_case(ct,ref)
            if True:
                rs.SetReferenceImage(ref)
                rs.SetDefaultPixelValue(-1000)
                rs.SetInterpolator(sitk.sitkLinear)
                ct_midres = rs.Execute(ct)
                x = sitk.GetArrayFromImage(ct_midres)
            x[x<-1024]=-1024 #fix for very low values
            out=np.zeros((zdim,xdim*2))
            out[:,0:xdim]=np.average(x,axis=1)
            out[:,xdim:]=np.average(x,axis=2)
            return out
        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            break

def load_batch(cases,aug_counter):
    batch_size=len(cases)
    p=Pool(processes=n_processes)
    #print(cases)
    #print('pool created')
    data=p.map(load_case,cases)
    #print('data returned')
    p.close()
    for i in range(batch_size):
        sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(data[i]),sitk.sitkInt16),join(out_dir,cases[i]+'---'+str(aug_counter).zfill(3)+'.nii.gz'))
        #print('File Written')

last_uid=uids[-1]  #loop to see what the highest count value is in the augmented case list
augmented_list=os.listdir(out_dir)
last_uid_count_list=[]
for f in augmented_list:
    if last_uid in f:
        last_uid_count_list.append(f)
if len(last_uid_count_list)<2: #needs to be at least validation case and one more...
    aug_counter=0
else:
    count_list = []
    for f in last_uid_count_list:
        try:
            aug_num=int(f.split('---')[-1].replace('.nii.gz',''))
            count_list.append(aug_num)
        except:
            a=0
    aug_counter=max(count_list)+1

if bake_validation:
    i=0
    for uid in uids:
        i+=1
        if i%10==0:
            print(i,'/',len(uids))
        x=load_val_case(uid)
        sitk.WriteImage(sitk.GetImageFromArray(x),join(out_dir,uid+'---val.nii.gz'))


#aug_counter=0
for i in range(n_augments_this_pass): 
    print('Augment:',aug_counter)
    print('Batch:',end=' ')
    for j in range(int(np.ceil(len(uids)/batch_size))):
        print(j,end=' ')
        start=j*batch_size
        end=(j+1)*batch_size
        if end>len(uids):
            end=len(uids)
        cases=uids[start:end]
        # current_batch_size=len(cases)
        # p=Pool(processes=n_processes)
        # data=p.map(load_case,cases)
        # p.close()
        # for k in range(current_batch_size):
        #     x=sitk.GetImageFromArray(data[k])
        #     fname=join(out_dir,cases[k]+'---'+str(aug_counter).zfill(3)+'.nii.gz')
        #     sitk.WriteImage(x,fname)
        load_batch(cases,aug_counter)
    aug_counter+=1
    print(' ')
    

    

