import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras as keras
import numpy as np
import os
from os.path import join
import SimpleITK as sitk
from scipy import ndimage
from numpy.random import rand
import datetime
global epoch_dir
from tensorflow.keras.callbacks import CSVLogger
from multiprocessing import Pool
import argparse
import sys
import random

parser=argparse.ArgumentParser(description='Keras Body Weight Eval')
parser.add_argument("model_name", type=str,help="prebuilt model name")
parser.add_argument("add_dropout", type=str,help="add dropout before final layer True/False, (0.5)")
parser.add_argument('model_loss',type=str,help='Model loss function (mse, mae)')
parser.add_argument('batch_size',type=int,help='Batch Size for training, eg 16')
args=parser.parse_args()
model_name=args.model_name
add_dropout_str=args.add_dropout
model_loss=args.model_loss
batch_size=int(args.batch_size)
if model_loss not in ['mse','mae']:
    print('Loss function should be mse or mae')
    sys.exit()
try:
    add_dropout=eval(add_dropout_str)
except:
    print("add_dropout must be either True or False")
    sys.exit()



"""
python Train_Carnie_v20_anymodel_noinit_256.py InceptionResNetV2 False mse 64

#runs InceptionResNetV2 with False flag for dropout layer and batch size of 64
"""

model_names=['DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet',
             'MobileNetV2', 'NASNetLarge', 'NASNetMobile', 'ResNet101', 'ResNet101V2', 'ResNet152',
             'ResNet152V2', 'ResNet50', 'ResNet50V2', 'VGG16', 'VGG19', 'Xception',
             'EfficientNetB0','EfficientNetB1','EfficientNetB2','EfficientNetB3','EfficientNetB4',
             'EfficientNetB5','EfficientNetB6','EfficientNetB7']

if model_name not in model_names:
    print('Unrecognized model name:',model_name)
    print('Must be in list:')
    print(model_names)
    sys.exit()



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D,Conv2D
from tensorflow.keras.layers import Reshape, Activation, Dense, Flatten,MaxPooling2D,Dropout,GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras import applications


def build_model(inp_shape,model_name,imagenet_weights=False,add_dropout=True):
    inputs=Input(shape=inp_shape)
    app_model=getattr(applications,model_name)
    if imagenet_weights:
        weights='imagenet'
    else:
        weights=None
    base=app_model(input_shape=inp_shape,weights=weights,include_top=False)
    x=GlobalAveragePooling2D()(base.layers[-2].get_output_at(0))
    if add_dropout:
        x=Dropout(0.5)(x)
    x = Dense(1, activation='linear')(x)
    model = Model(inputs=base.inputs, outputs=x)
    return model

n_epochs=1000
axial_dim=128
xdim=int(axial_dim)
ydim=int(axial_dim)
zdim=int(256)

inp_shape= (zdim,2*xdim,1)
model = build_model(inp_shape,model_name,imagenet_weights=False,add_dropout=add_dropout)

#batch_size=16
n_processes=16
xextent=float(500)
yextent=float(500)
zextent=float(1500)

spacing_string=str(int(xextent))+'_'+str(int(yextent))+'_'+str(int(zextent))+'-'+str(xdim)+'_'+str(ydim)+'_'+str(zdim)
df=pd.read_csv('weights_and_series_uids_v1_shuffle.csv',index_col=0)  #pre-shuffled case list so each run gets the same batch of training/validation cases
training_percent=90
#df = df.sample(frac=1).reset_index(drop=True) #randomly reshuffles dataframe
n_training=int(len(df)*float(training_percent/100.))
n_testing=len(df)-n_training
training_uids=df.SeriesUID.values[:n_training]
training_weights=df.PatientWeight.values[:n_training]
testing_uids=df.SeriesUID.values[n_training:]
testing_weights=df.PatientWeight.values[n_training:]

#pre-augmented CT files should be in 'ct_dir' foler named with the convention "[CASE]---[AUGMENT # or 'val' (unaugmented)].nii.gz"
#eg baked_training_256/Case001---005.nii.gz would be a case called 'Case001' and augment version number 5
#baked_training_256/Series050---val.nii.gz would be case called 'Series050' and the unaugmented version which is to be used for validation
#The series name (Case001 or Series050 in these examples will be cross referenced to the scalar value associated in the pandas csv loaded above

ct_dir='baked_training_256'
uids=df.SeriesUID.values
last_uid=uids[-1]  #loop to see what the highest count value is in the augmented case list
augmented_list=os.listdir(ct_dir)
last_uid_count_list=[]
for f in augmented_list:
    if last_uid in f:
        last_uid_count_list.append(f)
count_list = []
for f in last_uid_count_list:
    try:
        aug_num=int(f.split('---')[-1].replace('.nii.gz',''))
        count_list.append(aug_num)
    except:
        a=0
aug_counter=max(count_list)

fname=os.path.basename(__file__).replace('.py','')+"_"+spacing_string+"_"+model_name+"_"+add_dropout_str+"_"+model_loss+"_"+str(batch_size)
print(fname)

def new_dir(d):
    if not os.path.exists(d):
        os.mkdir(d)
    return
new_dir('logs')
new_dir('models')




log_dir="logs/"+fname + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


zeds=np.zeros((zdim,ydim,xdim))
xy_max_crop_dist=70 #mm
resample_extent=(xextent,yextent,zextent)  #max extent was 1942
final_resample_dims=(zdim,ydim,xdim)  #numpy zyx
final_resample_spacing=np.array((resample_extent[0]/final_resample_dims[2],resample_extent[1]/final_resample_dims[1],resample_extent[2]/final_resample_dims[0])) #sitk xyz
rss=final_resample_spacing

#@jit
def random_crop(image):
    np.random.seed(random.randint(0,65535))
    random_crop_pct=0.5
    if True:
    #try:
        #ar=sitk.GetArrayFromImage(image)
        spacing=image.GetSpacing()
        cropfilter=sitk.CropImageFilter()
        x_crop=int(xy_max_crop_dist/spacing[0]*rand())
        y_crop=int(xy_max_crop_dist/spacing[1]*rand())
        #midz=ar.shape[0]*np.random.rand()#ar.shape[0]/2
        midz = image.GetSize()[2] * np.random.rand()  # ar.shape[0]/2
        z_lower_crop=int(random_crop_pct*midz*rand())
        #z_upper_crop=int(random_crop_pct*(ar.shape[0]-midz)*rand())
        z_upper_crop = int(random_crop_pct * (image.GetSize()[2] - midz) * rand())
        cropfilter.SetLowerBoundaryCropSize([x_crop,y_crop, z_lower_crop])
        cropfilter.SetUpperBoundaryCropSize([x_crop,y_crop, z_upper_crop])
        image_crop = cropfilter.Execute(image)
        return image_crop
    # except Exception as e:
    #     print(image.GetSize(),e)
    #     return image#,label

#@jit
def augment_case(x,reference_im):
    np.random.seed(random.randint(0,65535))
    ct=x
    max_rotation_deg=5 #20.
    max_translation=7.  #25
    max_gauss_sigma=1.0
    max_hu_shift=30
    max_noise=100
    sharpening_range=0.6
    sharpening_alpha=0.5
    rotx=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
    roty=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
    rotz=(np.random.rand()-0.5)*max_rotation_deg*2*3.1415/360
    tx=(np.random.rand()-0.5)*max_translation
    ty=(np.random.rand()-0.5)*max_translation
    tz=(np.random.rand()-0.5)*max_translation
    sig=np.random.rand()*max_gauss_sigma
    sharp=sharpening_range*(1-0.3*(np.random.rand()-0.5))
    salpha=sharpening_alpha*(1-0.8*(np.random.rand()-0.5))
    hu_shift=(np.random.rand()-0.5)*max_hu_shift
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
    ar+=((np.random.random(ar.shape)-0.5)*max_noise).astype('int16')
    return ar.astype('float32')

#@jit
def load_case(label_name):
    while(True):
        try:
            np.random.seed(random.randint(0,65535))
            ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name+'---'+str(np.random.randint(0,aug_counter)).zfill(3)+'.nii.gz')),sitk.sitkInt16)
            x=sitk.GetArrayFromImage(ct)
            out=np.zeros((1,x.shape[0],x.shape[1],1))
            out[0,...,0]=x
            # out[0, ..., 1] = x
            # out[0, ..., 2] = x
            return out
        except (KeyboardInterrupt, SystemExit):
                print("Exiting...")
                break
    # while(True):
    #     try:
    #         ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name+'.nii.gz')),sitk.sitkInt16)
    #         rs=sitk.ResampleImageFilter()
    #         if True:
    #             ct=random_crop(ct) #applies random cropping to CT/Label combo
    #         origin=np.array(ct.GetOrigin())
    #         original_dims=np.array(ct.GetSize())
    #         original_spacing=np.array(ct.GetSpacing())
    #         original_extent=original_dims*original_spacing
    #         if True:
    #             origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
    #         origin[2]=origin[2]-origin_shift
    #         delta_extent=resample_extent-original_extent
    #         delta_x=delta_extent[0]/2.
    #         delta_y=delta_extent[1]/2.
    #         new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
    #         ref=sitk.GetImageFromArray(zeds)
    #         ref.SetSpacing(final_resample_spacing)
    #         ref.SetOrigin(new_origin)
    #         if True:
    #             x=augment_case(ct,ref)
    #         else:
    #             rs.SetReferenceImage(ref)
    #             rs.SetDefaultPixelValue(-1000)
    #             rs.SetInterpolator(sitk.sitkLinear)
    #             ct_midres = rs.Execute(ct)
    #             x = sitk.GetArrayFromImage(ct_midres)
    #         x[x<-1024]=-1024 #fix for very low values
    #         out=np.zeros((1,zdim,xdim*2,3))
    #         out[0,:,0:xdim,0]=np.average(x,axis=1)
    #         out[0,:,xdim:,0]=np.average(x,axis=2)
    #         out[0,:,0:xdim,1]=np.average(x,axis=1)
    #         out[0,:,xdim:,1]=np.average(x,axis=2)
    #         out[0,:,0:xdim,2]=np.average(x,axis=1)
    #         out[0,:,xdim:,2]=np.average(x,axis=2)
    #         return out
    #     except (KeyboardInterrupt, SystemExit):
    #         print("Exiting...")
    #         break



def load_val_case(label_name):
    while(True):
        try:
            ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name+'---val.nii.gz')),sitk.sitkInt16)
            x=sitk.GetArrayFromImage(ct)
            out=np.zeros((1,x.shape[0],x.shape[1],1))
            out[0,...,0]=x
            # out[0, ..., 1] = x
            # out[0, ..., 2] = x
            return out
        except (KeyboardInterrupt, SystemExit):
                print("Exiting...")
                break
    #
    # while(True):
    #     try:
    #         ct=sitk.Cast(sitk.ReadImage(join(ct_dir,label_name+'.nii.gz')),sitk.sitkInt16)
    #         rs=sitk.ResampleImageFilter()
    #         # if True:
    #         #     ct=random_crop(ct) #applies random cropping to CT/Label combo
    #         origin=np.array(ct.GetOrigin())
    #         original_dims=np.array(ct.GetSize())
    #         original_spacing=np.array(ct.GetSpacing())
    #         original_extent=original_dims*original_spacing
    #         if True:
    #             origin_shift=(0.5)*(resample_extent[2]-original_extent[2]) #puts in centre of slab
    #         origin[2]=origin[2]-origin_shift
    #         delta_extent=resample_extent-original_extent
    #         delta_x=delta_extent[0]/2.
    #         delta_y=delta_extent[1]/2.
    #         new_origin=np.array((origin[0]-delta_x,origin[1]-delta_y,origin[2]))
    #         ref=sitk.GetImageFromArray(zeds)
    #         ref.SetSpacing(final_resample_spacing)
    #         ref.SetOrigin(new_origin)
    #         # if False:
    #         #     x=augment_case(ct,ref)
    #         if True:
    #             rs.SetReferenceImage(ref)
    #             rs.SetDefaultPixelValue(-1000)
    #             rs.SetInterpolator(sitk.sitkLinear)
    #             ct_midres = rs.Execute(ct)
    #             x = sitk.GetArrayFromImage(ct_midres)
    #         x[x<-1024]=-1024 #fix for very low values
    #         out=np.zeros((1,zdim,xdim*2,3))
    #         out[0,:,0:xdim,0]=np.average(x,axis=1)
    #         out[0,:,xdim:,0]=np.average(x,axis=2)
    #         out[0,:,0:xdim,1]=np.average(x,axis=1)
    #         out[0,:,xdim:,1]=np.average(x,axis=2)
    #         out[0,:,0:xdim,2]=np.average(x,axis=1)
    #         out[0,:,xdim:,2]=np.average(x,axis=2)
    #         return out
    #     except (KeyboardInterrupt, SystemExit):
    #         print("Exiting...")
    #         break

#@jit(parallel=True,forceobj=True)
def load_batch(cases,weights,augment=True):
    batch_size=len(cases)
    p = Pool(processes=n_processes)
    x_batch=np.zeros((batch_size,zdim,xdim*2,1))
    y_batch=np.zeros((batch_size,1))
    if augment:
        data=p.map(load_case,cases)
    else:
        data=p.map(load_val_case,cases)
    p.close()
    for i in range(batch_size):
        #x=load_case(cases[i],augment=augment)
        x_case=data[i][0,...]
        #x_case=(x_case-x_case.min())/(x_case.max()-x_case.min()) #rescale [0-1]
        x_batch[i,...]=x_case
        y_batch[i,0]=weights[i]
    return x_batch,y_batch

def training_generator(training_cases,training_weights,batch_size):
    idx=0
    size=len(training_cases)
    while True:
        last_batch=idx+batch_size>size
        end = idx+batch_size if not last_batch else size
        if idx==end:
            idx=0
            end=batch_size
        x,y=load_batch(training_cases[idx:end],training_weights[idx:end],augment=True) #changed to false
        yield x.astype('float32'),y.astype('float32')
        if not last_batch:
            idx=end
        else:
            idx=0

def testing_generator(validation_cases,validation_weights,batch_size):
    idx_t=0
    size=len(validation_cases)
    while True:
        last_batch=idx_t+batch_size>size
        end = idx_t + batch_size if not last_batch else size
        if idx_t==end:
            idx_t=0
            end=batch_size
        x,y=load_batch(validation_cases[idx_t:end],validation_weights[idx_t:end],augment=False)
        yield x.astype('float32'),y.astype('float32')
        if not last_batch:
            idx_t=end
        else:
            idx_t=0

#if __name__ == '__main__':
if True:

    gen=training_generator(training_uids,training_weights,batch_size)
    gen_test=testing_generator(testing_uids,testing_weights,batch_size)
    print('X[0].shape: ',inp_shape)

    #adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-8,decay=0.0)
    #adam = keras.optimizers.Adam(lr=3e-4, decay=3e-4 / 200)
    adam=keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
    #model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse','mae','accuracy'])
    model.compile(optimizer=adam, loss=model_loss, metrics=['mse','mae','mean_absolute_percentage_error'])
    model.summary()
    checkpointer = ModelCheckpoint('models/'+fname+'.hdf5', save_best_only=True, mode='min', monitor='val_mse')
    csv_logger = CSVLogger('logs/'+fname+'.csv')
    callbacks=[checkpointer,csv_logger]
    history=model.fit(x=gen,validation_data=gen_test, epochs=n_epochs, callbacks=callbacks,steps_per_epoch=int(np.ceil(n_training/batch_size)),validation_steps=int(np.ceil(n_testing/batch_size)))  #(available_cases-validation_size)
    print(history.history.keys())
    hist_df = pd.DataFrame(history.history)
##    hist_df.to_csv(fname+'.csv')

