from Core import getMCAD_X,getADNI_X
import pandas as pd
import xlrd
import numpy as np
from Core.subtypeUtils import setting,regress_cov
import neuroCombat
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


def outputADNI_Ex_List(table):
    ADNI_extra = pd.read_csv(r"E:\brain\subtype\subtype-data\ADNI_extra_info.csv")
    for index, row in table.iterrows():
        res = ADNI_extra.loc[(ADNI_extra['PTID'] == row['PTID']) & (ADNI_extra['VISCODE3'] == row['VISCODE3']) & (row['DataFrom']!=1)]
        if(res.shape[0] > 0):
            print('G:/ADNI_fMRI/precessed_fmri/'+row['PTID']+"_"+row['VISCODE3'])

def findExtra(table):
    ADNI_extra = pd.read_csv(r"E:\brain\subtype\subtype-data\ADNI_extra_info.csv")
    for index, row in table.iterrows():
        res = ADNI_extra.loc[(ADNI_extra['PTID'] == row['PTID']) & (ADNI_extra['VISCODE3'] == row['VISCODE3'])]
        if(res.shape[0] > 0):
            table.loc[index,'DataFrom'] = 1
            table.loc[index,'Subject'] = table.loc[index,'PTID']+'_'+table.loc[index,'VISCODE3']
    return table
def MCAD_Cal_X():
    corrPath = setting.corrPath
    centerNames = setting.centerNames
    imageNames = setting.imageNames
    group = setting.group
    centers = setting.centers
    NC_data,NC_age,NC_gender, NC_center = getMCAD_X(imageNames=imageNames,centerNames=centerNames,group=group,group_id=1,corrPath=corrPath,centers=centers,isNorm=False)
    MCI_data, MCI_age, MCI_gender, MCI_center= getMCAD_X(imageNames=imageNames,centerNames=centerNames,group=group,group_id=2,corrPath=corrPath,centers=centers,isNorm=False)
    AD_data, AD_age, AD_gender, AD_center = getMCAD_X(imageNames=imageNames,centerNames=centerNames,group=group,group_id=3,corrPath=corrPath,centers=centers,isNorm=False)
    MCAD_DATA = np.concatenate([NC_data.T,MCI_data.T,AD_data.T],axis=0)
    MCAD_AGE = np.concatenate([NC_age,MCI_age,AD_age],axis=0)
    MCAD_GENDER = np.concatenate([NC_gender,MCI_gender,AD_gender],axis=0)
    MCAD_CENTER = np.concatenate([NC_center,MCI_center,AD_center],axis=0)

    MCAD_LABEL = np.concatenate([[1 for i in range(NC_data.shape[1])],[2 for i in range(MCI_data.shape[1])],[3 for i in range(AD_data.shape[1])]],axis=0)
    infoTable = pd.DataFrame(np.concatenate([MCAD_AGE.reshape([-1,1]),MCAD_GENDER.reshape([-1,1]),MCAD_CENTER.reshape([-1,1]),MCAD_LABEL.reshape([-1,1])],axis=1),columns=['AGE','GENDER','batch','GROUP'])
    MCAD_DATA = neuroCombat.neuroCombat(MCAD_DATA.T,covars=infoTable,batch_col='batch',categorical_cols=['GENDER','GROUP'],continuous_cols=['AGE'])
    NC_data = MCAD_DATA[:,:NC_data.shape[1]]
    MCI_data = MCAD_DATA[:,NC_data.shape[1]:NC_data.shape[1]+MCI_data.shape[1]]
    AD_data = MCAD_DATA[:,AD_data.shape[1]:]

    np.save('./ADATA/MCAD_NC.npy',[NC_data,NC_age,NC_gender])
    np.save('./ADATA/MCAD_MCI.npy', [MCI_data, MCI_age, MCI_gender])
    np.save('./ADATA/MCAD_AD.npy', [AD_data, AD_age, AD_gender])


    return MCAD_DATA
def ADNI_Cal_X(Stage=1):
    AD_Define_Bl = pd.read_csv('./table/AD_Define_Bl_with_cluster.csv')
    AD_Define_Bl = pd.read_csv('./table/AD_ABETA.csv')

    MCI_Define_Bl = pd.read_csv('./table/MCI_Define_Bl_with_cluster.csv')
    MCI_Define_Bl = pd.read_csv('./table/MCI_ABETA.csv')
    NC_Define_Bl = pd.read_csv('./table/NC_Define_Bl_All.csv')
    NC_Define_Bl = pd.read_csv('./table/NC_ABETA.csv')


    AD_Define_Bl = findExtra(AD_Define_Bl)
    MCI_Define_Bl = findExtra(MCI_Define_Bl)
    NC_Define_Bl = findExtra(NC_Define_Bl)




    NC_data = getADNI_X(NC_Define_Bl,isNorm=False)
    NC_Define_Bl['AGE2'] = NC_Define_Bl['AGE'] + NC_Define_Bl['Years_bl']
    NC_AGE = NC_Define_Bl['AGE2'].values
    NC_SITE = NC_Define_Bl['SITE'].values
    NC_GENDER = NC_Define_Bl['PTGENDER'].map({'Male':1,'Female':2})

    MCI_data = getADNI_X(MCI_Define_Bl,isNorm=False)
    MCI_AGE = MCI_Define_Bl['AGE2'].values
    MCI_SITE = MCI_Define_Bl['SITE'].values
    MCI_GENDER = MCI_Define_Bl['PTGENDER'].map({'Male':1,'Female':2})

    AD_data = getADNI_X(AD_Define_Bl,isNorm=False)
    AD_AGE = AD_Define_Bl['AGE2'].values
    AD_SITE = AD_Define_Bl['SITE'].values
    AD_GENDER = AD_Define_Bl['PTGENDER'].map({'Male':1,'Female':2})






    ADNI_DATA = np.concatenate([NC_data,MCI_data,AD_data,Long_data],axis=1)
    ADNI_AGE = np.concatenate([NC_AGE,MCI_AGE,AD_AGE,Long_AGE],axis=0)
    ADNI_GENDER = np.concatenate([NC_GENDER,MCI_GENDER,AD_GENDER,Long_GENDER],axis=0)
    ADNI_SITE = np.concatenate([NC_SITE,MCI_SITE,AD_SITE,Long_SITE],axis=0)
    ADNI_SITE = ADNI_SITE.astype(np.int)
    ADNI_LABEL = np.concatenate([[1 for i in range(NC_data.shape[1])],
                                 [2 for i in range(MCI_data.shape[1])],
                                 [3 for i in range(AD_data.shape[1])],
                    
                                 ],axis=0)

    infoTable = pd.DataFrame(np.concatenate([ADNI_AGE.reshape([-1,1]),ADNI_GENDER.reshape([-1,1]),ADNI_SITE.reshape([-1,1]),ADNI_LABEL.reshape([-1,1])],axis=1),columns=['AGE','GENDER','batch','GROUP'])


    sns.heatmap(ADNI_DATA,vmin=ADNI_DATA.min(),vmax=ADNI_DATA.max(),cmap=sns.color_palette('coolwarm'))
    plt.show()

    infoTable.to_csv('./ADATA/ADNI_INFO.csv',index=None)
    pd.DataFrame(ADNI_DATA,index=None).to_csv('./ADATA/ADNI.csv',index=None)

    robjects.r.source('./R/combat.R')

    ADNI_DATA = np.loadtxt('./ADATA/ADNI_Combat.txt')


    NC_data = ADNI_DATA[:,:NC_data.shape[1]]
    MCI_data = ADNI_DATA[:,NC_data.shape[1]:NC_data.shape[1]+MCI_data.shape[1]]
    p = NC_data.shape[1]+MCI_data.shape[1]
    AD_data = ADNI_DATA[:,p:p+AD_data.shape[1]]
    p = p + AD_data.shape[1]


    np.save('./ADATA/ADNI_NC_ABETA.npy',[NC_data,NC_AGE,NC_GENDER])
    np.save('./ADATA/ADNI_MCI_ABETA.npy',[MCI_data,MCI_AGE,MCI_GENDER])
    np.save('./ADATA/ADNI_AD_ABETA.npy',[AD_data,AD_AGE,AD_GENDER])


if __name__ == '__main__':
    ADNI_Cal_X()
    MCAD_Cal_X()