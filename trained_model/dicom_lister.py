import SimpleITK as sitk
import sys

import pandas as pd
import os
from os.path import join
import pydicom as dicom



def get_pydicom_uids(in_dir):
    uids=[]
    for f in os.listdir(in_dir):
        if f.endswith('.dcm'):
            uid=dicom.read_file(join(in_dir,f))
            if uid not in uids:
                uids.append(uid)
    return uids

def nested_dir_to_df(in_dir):
    reader=sitk.ImageSeriesReader()
    df=pd.DataFrame(columns=['Patient','SeriesDescription','SeriesUID','Modality','FrameOfReferenceUID','Filenames'])
    counter=0
    for root,dirs,files in os.walk(in_dir):
        for d in dirs:
            #if any(fname.endswith('.dcm') for fname in os.listdir(join(root,d))):
            if True:
                uids=reader.GetGDCMSeriesIDs(join(root,d))
                for uid in uids:
                    print(uid)
                    filenames=reader.GetGDCMSeriesFileNames(join(root,d),uid)
                    dcm=dicom.read_file(filenames[0])
                    modality=str(dcm.Modality)
                    series_uid=str(dcm.SeriesInstanceUID)
                    patient_name=str(dcm.PatientName)
                    description=str(dcm.SeriesDescription)
                    for_uid=str(dcm.FrameOfReferenceUID)
                    #if modality=='CT' and uid not in df.SeriesUID.values: #and uid not in df.SeriesUIDs?
                    if uid not in df.SeriesUID.values: #and uid not in df.SeriesUIDs?
                        #print('CT')
                        df.loc[counter]=[patient_name,description,series_uid,modality,for_uid,filenames]
                        counter+=1
        #if any(fname.endswith('.dcm') for fname in os.listdir(root)):
        if True:
            uids=reader.GetGDCMSeriesIDs(root)
            for uid in uids:
                print(uid)
                filenames=reader.GetGDCMSeriesFileNames(root,uid)
                dcm=dicom.read_file(filenames[0])
                modality=str(dcm.Modality)
                series_uid=str(dcm.SeriesInstanceUID)
                patient_name=str(dcm.PatientName)
                description=str(dcm.SeriesDescription)
                for_uid=str(dcm.FrameOfReferenceUID)
                #if modality=='CT' and uid not in df.SeriesUID.values: #and uid not in df.SeriesUIDs?
                if uid not in df.SeriesUID.values: #and uid not in df.SeriesUIDs?
                    #print('CT')
                    df.loc[counter]=[patient_name,description,series_uid,modality,for_uid,filenames]
                    counter+=1
    return df



    
