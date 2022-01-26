import os
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from .subtypeUtils import setting,to_categorical,regress_cov,altered_fc_idx,calCorr

def getMCAD_X_Norm(Group_id,ref_id=None,is_Origin=False):
    PATHS = ['E:/brain/subtype/src/ADATA/MCAD_NC.npy',
             'E:/brain/subtype/src/ADATA/MCAD_MCI.npy',
             'E:/brain/subtype/src/ADATA/MCAD_AD.npy',]
    
    dataMatrix = np.load(PATHS[Group_id-1],allow_pickle=True)[0]
    if(is_Origin):
        return dataMatrix
    if(ref_id == None):
        ref = dataMatrix
    else:
        ref = np.load(PATHS[ref_id-1],allow_pickle=True)[0]

    for i in range(dataMatrix.shape[0]):
        aMin = np.mean(ref[i, ref[i, :].argsort()[:3]])
        aMax = np.mean(ref[i, ref[i, :].argsort()[-3:]])

        dataMatrix[i, :] = (dataMatrix[i, :] - aMin) / (aMax - aMin)

    dataMatrix[dataMatrix < 0] = 0

    return dataMatrix
def getADNI_X_Norm(Group_id,ref_id=None,is_Origin=False):
    PATHS = ['./ADATA/ADNI_NC_ABETA.npy',
            './ADATA/ADNI_MCI_ABETA.npy',
            './ADATA/ADNI_AD_ABETA.npy',
            './ADATA/ADNI_Con.npy',
            './ADATA/ADNI_Long.npy',
             './ADATA/MCI_Long.npy'
            ]
    dataMatrix = np.load(PATHS[Group_id-1],allow_pickle=True)[0]
    if(is_Origin):
        return dataMatrix
    if(ref_id == None):
        ref = dataMatrix
    else:
        ref = np.load(PATHS[ref_id-1],allow_pickle=True)[0]

    for i in range(dataMatrix.shape[0]):
        #aMin = np.mean(ref[i, ref[i, :].argsort()[:3]])
        #aMax = np.mean(ref[i, ref[i, :].argsort()[-3:]])

        aMin = np.min(ref[i,:])
        aMax = np.max(ref[i,:])

        dataMatrix[i, :] = (dataMatrix[i, :] - aMin) / (aMax - aMin)

    dataMatrix[dataMatrix < 0] = 0

    return dataMatrix

def calMatrix(imagePath, atlas, atlasIndex=None):
    '''
    由atlas提取时间序列后计算功能连接矩阵
    '''
    if atlasIndex is None:
        atlasIndex = [i for i in range(1, np.max(atlas) + 1)]
    data = nib.load(imagePath).get_fdata()
    data = np.nan_to_num(data)
    tsMatrix = np.zeros([data.shape[3], 263])
    for i, index in enumerate(atlasIndex):
        # if i == 255:
        #    p = i + 1
        #    tsMatrix[:,i-1] = np.mean(data[atlas == p,:],axis=0)
        # print(index,'有'+str(np.shape(np.where(atlas == index))[1]),'个voxel')
        tsMatrix[:, i] = np.mean(data[atlas == index, :], axis=0)

    pearsonData = np.corrcoef(tsMatrix, rowvar=False)
    return pearsonData

def getMCAD_X(imageNames,centerNames,group,group_id,corrPath,centers,AGE=setting.age,GENDER=setting.gender,getMAX_MIN=False,MAX_MIN_id=3,isNorm=True):

    dataMatrix = np.zeros([altered_fc_idx.shape[0], imageNames[group == group_id].shape[0]])
    sampleP = 0

    regAge = []
    regGender = []
    regCenter = []

    for centerP, center in enumerate(centerNames):
        centerPath = os.path.join(corrPath, center)
        centerImageNames = imageNames[(centers == (centerP + 1)) & (group == group_id)]
        regAge.extend(AGE[(centers == (centerP + 1)) & (group == group_id)].tolist())
        regGender.extend(GENDER[(centers == (centerP + 1)) & (group == group_id)].tolist())
        regCenter.extend([centerP for i in range(centerImageNames.shape[0])])

        for imageName in centerImageNames:
            filePath = os.path.join(centerPath, imageName + '_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
            dataMatrix[:, sampleP] = corrData[tuple(altered_fc_idx.transpose().tolist())]
            sampleP += 1
            print(imageName)

    if not isNorm:
        return dataMatrix,regAge,regGender,regCenter

    onehotCenter = to_categorical(regCenter, num_classes=7)

    if(group_id==MAX_MIN_id):
        MCAD_MAX_MIN = np.zeros([216, 2])
        for i in range(dataMatrix.shape[0]):
            dataMatrix[i,:] = regress_cov(dataMatrix[i,:].reshape([-1,1]),np.concatenate((np.array(regAge).reshape([-1,1]), np.array(regGender).reshape([-1,1]), onehotCenter), axis=1), center=False, keep_scale=False).squeeze()

            aMin = np.mean(dataMatrix[i, dataMatrix[i, :].argsort()[:3]])
            aMax = np.mean(dataMatrix[i, dataMatrix[i, :].argsort()[-3:]])

            dataMatrix[i, :] = (dataMatrix[i, :] - aMin) / (aMax - aMin)
            dataMatrix[i, :] = (dataMatrix[i, :] - dataMatrix[i, :].min()) / (
                    dataMatrix[i, :].max() - dataMatrix[i, :].min())

            MCAD_MAX_MIN[i, 1] = aMin
            MCAD_MAX_MIN[i, 0] = aMax

        dataMatrix[dataMatrix < 0] = 0
        if(getMAX_MIN):
            return MCAD_MAX_MIN
        return dataMatrix
    else:
        MCAD_MAX_MIN = getMCAD_X(imageNames,centerNames,group,MAX_MIN_id,corrPath,centers,getMAX_MIN=True,MAX_MIN_id=MAX_MIN_id)
        for i in range(dataMatrix.shape[0]):
            dataMatrix[i, :] = regress_cov(dataMatrix[i, :].reshape([-1,1]), np.concatenate((np.array(regAge).reshape([-1,1]), np.array(regGender).reshape([-1,1]), onehotCenter), axis=1),
                                           center=False, keep_scale=False).squeeze()
            dataMatrix[i, :] = (dataMatrix[i, :] - MCAD_MAX_MIN[i, 1]) / (MCAD_MAX_MIN[i, 0] - MCAD_MAX_MIN[i, 1])
        dataMatrix[dataMatrix < 0] = 0
        return dataMatrix

def getADNI_X(AD_Define_Bl,getMAX_MIN=False,isNorm=True):
    Pathes = [r"E:\brain\subtype\subtype-data\ADNI\ADNI_ex\mean_ts"
        , r"D:\DATA\ADNI\ADNI_BNAtlas_mean_ts",
              r"H:\subtype-data\ADNI\FC\roi2roi_r_pearson_correlation"]
    imageNames = AD_Define_Bl['Subject'].values
    dataFrom = AD_Define_Bl['DataFrom'].values

    SITE = AD_Define_Bl['SITE'].values
    Gender = AD_Define_Bl['PTGENDER'].map({'Male':1,'Female':2}).values
    Age = AD_Define_Bl['AGE'].values

    dataMatrix = np.zeros([altered_fc_idx.shape[0], imageNames.shape[0]])
    sampleP = 0
    for imageName, source in zip(imageNames, dataFrom):
        if (source == 1):
            filePath = os.path.join(Pathes[0], imageName + '_ts.csv')
            corrData = calCorr(filePath)
        elif (source == 2):
            filePath = os.path.join(Pathes[1], imageName + '_ts.csv')
            corrData = calCorr(filePath)
        elif (source == 3):
            print(imageName)
            filePath = os.path.join(Pathes[2], imageName + '_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
        dataMatrix[:, sampleP] = corrData[tuple(altered_fc_idx.transpose().tolist())]
        sampleP += 1
    if(not isNorm):
        return dataMatrix
    ADNI_MAX_MIN = np.zeros([216, 2])
    for i in range(dataMatrix.shape[0]):
        md = sm.MixedLM(dataMatrix[i,:],
                   np.concatenate((np.array(Age).reshape([-1, 1]), np.array(Gender).reshape([-1, 1])), axis=1),
                   groups=SITE)
        mdf = md.fit()
        dataMatrix[i, :] = mdf.resid

        aMin = np.mean(dataMatrix[i, dataMatrix[i, :].argsort()[:3]])
        aMax = np.mean(dataMatrix[i, dataMatrix[i, :].argsort()[-3:]])
        if isNorm:
            dataMatrix[i, :] = (dataMatrix[i, :] - aMin) / (aMax - aMin)
        ADNI_MAX_MIN[i, 1] = aMin
        ADNI_MAX_MIN[i, 0] = aMax
    if isNorm:
        dataMatrix[dataMatrix < 0] = 0
    np.save('data/ADNI_MAX_MIN_All_withoutMB.npy',ADNI_MAX_MIN)
    # VolumeData = np.load('./AD_All_volume_120.npy')
    # VolumeData = VolumeData.T

    return dataMatrix


