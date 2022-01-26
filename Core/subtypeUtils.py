import numpy as np
import pandas as pd
import os 
import nibabel as nib
from scipy.stats import f_oneway
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
from sklearn import linear_model
import xlrd
import seaborn as sns
import matplotlib.pyplot as plt
altered_fc_idx = np.loadtxt('./altered_fc_idx.txt')
altered_fc_idx = altered_fc_idx.astype(np.int)
altered_fc_idx = altered_fc_idx - 1
class setting:

    corrPath = 'D:/DATA/MCAD/MCAD_BN_Atlas_corr'
    centerNames = ['HH_W', 'PL_G', "PL_S", "QL_W", 'XW_H', 'XW_Z', 's07']

    excelfile = xlrd.open_workbook(r"E:\brain\subtype\src\mcad_info_687_del670.xlsx")
    sheet = excelfile.sheet_by_name('Sheet1')
    imageNames = np.array(sheet.col_values(0)[1:])
    group = np.array(sheet.col_values(2)[1:])
    centers = np.array(sheet.col_values(6)[1:])

    gender = np.array(sheet.col_values(3)[1:])
    age = np.array(sheet.col_values(4)[1:])

    #ADNI_CLuster_Seq = [1,2,3,0]
    ADNI_CLuster_Seq = [0,2,1,3]


def calCorr(Path):
    data = pd.read_csv(Path,header=None,index_col=None)
    data = data.to_numpy()
    pearsonData = np.corrcoef(data, rowvar=0)
    for i in range(data.shape[1]):
        if((np.sum(np.abs(data[:,i])) < 0.01)):
            print('TS file error:',Path)
    #np.savetxt('3_AD212_corr_r.txt',pearsonData,delimiter=',',fmt="%.5f")
    return pearsonData

def ADNI_Cluster_MCI(MCI_Info, weight, ADNI_MAX_MIN, altered_fc_idx):
    imageNames = MCI_Info['PTID'].values
    subject = MCI_Info['Subject'].values
    dataFrom = MCI_Info['DataFrom'].values

    Pathes = [r"E:\brain\subtype\subtype-data\ADNI\ADNI_ex\roi2roi_r_pearson_correlation",
              r"D:\DATA\ADNI_BNAtlas_mean_ts",
              r"H:\subtype-data\ADNI\FC\roi2roi_r_pearson_correlation"]

    dataMatrix = np.zeros([altered_fc_idx.shape[0], imageNames.shape[0]])
    sampleP = 0
    MCI_cluster = []
    MCI_H = []
    for imageName, source in zip(subject, dataFrom):
        if(source == 1):
            filePath = os.path.join(Pathes[0], imageName+'_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
        elif(source == 2):
            filePath = os.path.join(Pathes[1], imageName+'_ts.csv')
            corrData = calCorr(filePath)
        elif(source == 3):
            filePath = os.path.join(Pathes[2], imageName+'_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
        dataMatrix[:, sampleP] = corrData[tuple(altered_fc_idx.transpose().tolist())]
        sampleP += 1

    for i in range(dataMatrix.shape[0]):
        dataMatrix[i, :] = (dataMatrix[i, :] - ADNI_MAX_MIN[i,1]) / (ADNI_MAX_MIN[i,0] - ADNI_MAX_MIN[i,1])

    for p in range(dataMatrix.shape[1]):
        feature = dataMatrix[:, p]
        distence,clusterRes = distenceMeasure(weight,feature,method='reg')
        MCI_H.append(distence)
        MCI_cluster.append(clusterRes)
    return MCI_cluster,MCI_H

def distenceMeasure(weight,feature,method = 'norm',isIntercept=False):
    if(method == 'norm'):
        normList = np.zeros([4,])
        for i in range(weight.shape[1]):
            normList[i] = np.linalg.norm(feature - weight[:,i])
        return normList,normList.argmin()
    elif(method == 'reg'):
        if(isIntercept):
            weight = np.hstack([weight, np.ones([weight.shape[0], 1])])
        h = np.dot(np.linalg.pinv(weight),feature)
        h = h[:4]
        return h,h.argmin()
    elif(method == 'rreg'):
        if (isIntercept):
            weight = np.hstack([weight, np.ones([weight.shape[0], 1])])
        feature = feature.reshape([-1,1])
        clf = linear_model.Ridge(fit_intercept=False,alpha=0.15)
        # 参数矩阵，即每一个alpha对于的参数所组成的矩阵
        # coefs = []
        # 根据不同的alpha训练出不同的模型参数
        clf.fit(weight,feature)
        print(clf.coef_)
        return clf.coef_,clf.coef_.argmin()

def ADNI_Cluster_MCI_by_Mean(MCI_Info,AD_Define_Bl):
    Pathes = [r"E:\brain\subtype\subtype-data\ADNI\ADNI_ex\mean_ts"
        , r"D:\DATA\ADNI_BNAtlas_mean_ts",
              r"H:\subtype-data\ADNI\FC\roi2roi_r_pearson_correlation"]
    cluster = AD_Define_Bl['cluster'].values
    altered_fc_idx = np.loadtxt(r"E:\brain\subtype\subtype-data\altered_fc_idx.txt")
    altered_fc_idx = altered_fc_idx.astype(np.int)
    altered_fc_idx = altered_fc_idx - 1

    imageNames = AD_Define_Bl['Subject'].values
    dataFrom = AD_Define_Bl['DataFrom'].values
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

    meanFc = np.zeros([216,4])
    for i in range(4):
        meanFc[:,i] = np.mean(dataMatrix[:,cluster == i],axis=1)

    imageNames = MCI_Info['PTID'].values
    subject = MCI_Info['Subject'].values
    dataFrom = MCI_Info['DataFrom'].values

    dataMatrix = np.zeros([altered_fc_idx.shape[0], imageNames.shape[0]])
    sampleP = 0
    MCI_cluster = []
    MCI_H = []
    for imageName, source in zip(subject, dataFrom):
        if(source == 1):
            filePath = os.path.join(Pathes[0], imageName+'_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
        elif(source == 2):
            filePath = os.path.join(Pathes[1], imageName+'_ts.csv')
            corrData = calCorr(filePath)
        elif(source == 3):
            filePath = os.path.join(Pathes[2], imageName+'_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
        dataMatrix[:, sampleP] = corrData[tuple(altered_fc_idx.transpose().tolist())]
        sampleP += 1

    for p in range(dataMatrix.shape[1]):
        feature = dataMatrix[:, p]
        distence,clusterRes = distenceMeasure(meanFc,feature,method='norm')
        MCI_H.append(distence)
        MCI_cluster.append(clusterRes)
    return MCI_cluster,MCI_H


def washData(infoTable, ADNI_MERGE):
    '''
    find the absence Gender and AGE
    '''
    for index, row in infoTable.iterrows():
        if(pd.isna(row['PTGENDER'])):
            res = ADNI_MERGE.loc[ADNI_MERGE['PTID'] == row['PTID']]
            res = res[pd.notna(res)]
            if(res.shape[0] > 0):
                infoTable.loc[index, 'PTGENDER'] = res['PTGENDER'].iloc[0]
            else:
                print('PTGENDER Not Found Any Info', row['PTID'])

    for index, row in infoTable.iterrows():
        if(pd.isna(row['Years_bl'])):
            res = ADNI_MERGE.loc[ADNI_MERGE['PTID'] == row['PTID']]
            res = res[pd.notna(res)]
            if(res.shape[0] > 0):
                infoTable.loc[index, 'Years_bl'] = res['Years_bl'].iloc[0]
            else:
                infoTable.loc[index, 'Years_bl'] = 0
                print('Years_bl Not Found Any Info', row['PTID'])

    
    return infoTable


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

import scipy.io as scio
import glob

def getADNI_TIV(AD_Define_Bl):
    TIVs = []
    relCSFs = []
    relGMVs = []
    relWMVs = []
    for index,row in AD_Define_Bl.iterrows():
        PTID = row['PTID']
        pattern = 'D:/DATA/ADNI_T1_2/report/cat*'+PTID+'*.mat'
        pathes = glob.glob(pattern)
        if(len(pathes) > 1):
            print(row['PTID'])
            print('------------------------------------------------')
        if(len(pathes) == 0):
            print(row['PTID'],row['DATE'],row['EXAMDATE'])
            print(pattern)
            continue
        #print(PTID,end=':')
        s = scio.loadmat(pathes[0],simplify_cells=True)
        TIV = s['S']['subjectmeasures']['vol_TIV']
        relCSF = s['S']['subjectmeasures']['vol_rel_CGW'][0]
        relGMV = s['S']['subjectmeasures']['vol_rel_CGW'][1]
        relWMV = s['S']['subjectmeasures']['vol_rel_CGW'][2]
        
        TIVs.append(TIV)
        relCSFs.append(relCSF)
        relGMVs.append(relGMV)
        relWMVs.append(relWMV)
        
        pattern = 'D:/DATA/ADNI_T1_2/ADNI*'+PTID+'*.nii'
        pathes = glob.glob(pattern)

    return TIVs,relCSFs,relGMVs,relWMVs

from scipy.linalg import lstsq
from sklearn.preprocessing import MinMaxScaler
def regress_cov(features, covariance, center=True, keep_scale=True):
    """
    :param covarance: covariance
    :param center: Boolean, wether to add intercept
    """
    if features.ndim == 1:
        features = features.reshape(len(features), 1)
    if covariance.ndim == 1:
        covariance = covariance.reshape(len(features), 1)
    residuals = np.zeros_like(features)
    result = np.zeros_like(features)
    if center is True:
        b = covariance
    else:
        b = np.hstack([covariance, np.ones([covariance.shape[0], 1])])
    for f in range(features.shape[1]):
        w = lstsq(b, features[:, f])[0]
        if center is True:
            residuals[:, f] = features[:, f] - covariance.dot(w)
        else:
            residuals[:, f] = features[:, f] - covariance.dot(w[:-1])
    if keep_scale is True:
        for f in range(features.shape[1]):
            if np.min(features[:, f]) == np.max(features[:, f]):
                result[:, f] = features[:, f]
            else:
                result[:, f] = MinMaxScaler(feature_range=(np.min(features[:, f]), np.max(features[:, f]))). \
                    fit_transform(residuals[:, f].reshape(-1, 1)).flatten()
    else:
        result = residuals
    return result

columns = ['AGE2','PTGENDER2',
 'APOE4',
 'FDG',
 'PIB',
 'AV45',
 'TAU',
 'PTAU',
 'CDRSB',
 'ADAS11',
 'ADAS13',
 'ADASQ4',
 'MMSE',
 'RAVLT_immediate',
 'RAVLT_learning',
 'RAVLT_forgetting',
 'RAVLT_perc_forgetting',
 'LDELTOTAL',
 'DIGITSCOR',
 'TRABSCOR',
 'FAQ',
 'MOCA',
 'EcogPtMem',
 'EcogPtLang',
 'EcogPtVisspat',
 'EcogPtPlan',
 'EcogPtOrgan',
 'EcogPtDivatt',
 'EcogPtTotal',
 'EcogSPMem',
 'EcogSPLang',
 'EcogSPVisspat',
 'EcogSPPlan',
 'EcogSPOrgan',
 'EcogSPDivatt',
 'EcogSPTotal',
 'Ventricles',
 'Hippocampus',
 'WholeBrain',
 'Entorhinal',
 'Fusiform',
 'MidTemp',
 'ICV',
 'mPACCdigit',
 'mPACCtrailsB',]


def cal_paramter_ANOVA_fdr(nodalData, col_num, prefix, age, gender, onehotCenter, cluster, path, group, groupIndex,
                           nii_path=r"E:\brain\BNatlas\BN_Atlas_6_centers.nii", save=False):
    '''
    这里是尝试看看图论里面的指标有没有差异
    '''
    print(nodalData.shape)
    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    rois = np.unique(template[:])[1:]

    f = np.zeros((col_num,))
    p = np.zeros((col_num,))
    img1 = np.zeros_like(template)
    img2 = np.zeros_like(template)
    for k, roi in zip(range(col_num), rois):
        var = nodalData[:, k]
        var = regress_cov(var, np.concatenate([age, gender, onehotCenter], axis=1), center=False, keep_scale=False)
        var = var[group == groupIndex]
        f[k], p[k] = f_oneway(var[cluster == 0], var[cluster == 1], var[cluster == 2], var[cluster == 3])
    # print(p)
    _, p_fdr = fdrcorrection(p, 0.05)
    # print(p_fdr)
    for k, roi in zip(range(col_num), rois):
        if p_fdr[k] <= 0.05:
            # print("pass")
            img1[template == roi] = f[k]
            img2[template == roi] = np.sign(f[k])

    nii1 = nib.Nifti1Image(img1, nii.affine)
    if (save):
        nib.save(nii1, path + prefix + '_ST' + '_anova_fdr0.05.nii')
    return p_fdr, f


def cal_paramter_ttest_map_posthoc(nodalData, col_index, prefix, age, gender, onehotCenter, cluster, group, groupIndex,
                                   path, nii_path=r"E:\brain\BNatlas\BN_Atlas_6_centers.nii", save=False):
    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    Tmap = []
    Pmap = []
    for i in range(3):
        for j in range(i + 1, 4):
            t = np.zeros((len(col_index),))
            p = np.zeros((len(col_index),))
            img1 = np.zeros_like(template)
            img2 = np.zeros_like(template)
            for k, roi in enumerate(list(col_index)):
                var = nodalData[:, k]

                var = regress_cov(var, np.concatenate([age, gender, onehotCenter], axis=1), center=False,
                                  keep_scale=False)
                var = var[group == groupIndex]
                t[k], p[k] = stats.ttest_ind(var[cluster == i], var[cluster == j])

            # _,p_fdr=fdrcorrection(p,0.05)
            # print(p_fdr)
            for k, roi in enumerate(list(col_index)):
                if (p[k] <= 0.05):
                    # print("pass")
                    img1[template == (roi + 1)] = t[k]
                    img2[template == (roi + 1)] = np.sign(t[k])
            nii1 = nib.Nifti1Image(img1, nii.affine)
            nii2 = nib.Nifti1Image(img2, nii.affine)
            if save:
                nib.save(nii1, path + prefix + '_ST' + str(i + 1) + str(j + 1) + '_ttest_fdr0.05.nii')
            # nib.save(nii2, path + prefix + '_ST' + str(i + 1) + str(j + 1) + '_ttest_fdr0.05_h.nii')
            Tmap.append(t)
            Pmap.append(p.copy())
    return np.array(Tmap), np.array(Pmap)

def cal_cluster_ttest_map_posthoc(sheet,col_index,prefix,age,gender,onehotCenter,cluster,group,groupIndex,path,nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii",save=False):
    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    Tmap = []
    Pmap = []
    for i in range(3):
        for j in range(i + 1, 4):
            t = np.zeros((len(col_index),))
            p = np.zeros((len(col_index),))
            img1 = np.zeros_like(template)
            img2 = np.zeros_like(template)
            for k, roi in enumerate(list(col_index)):
                var = np.array(sheet.col_values(roi)[1:])
                
                var = regress_cov(var,np.concatenate([age,gender,onehotCenter],axis=1),center=False,keep_scale=False)
                var = var[group == groupIndex]
                t[k], p[k] = stats.ttest_ind(var[cluster == i], var[cluster == j])

            #_,p_fdr=fdrcorrection(p,0.05)
            #print(p_fdr)
            for k, roi in enumerate(list(col_index)):
                if (p[k]<=0.05):
                    #print("pass")
                    img1[template == (roi+1)] = t[k]
                    img2[template == (roi+1)] = np.sign(t[k])
            nii1 = nib.Nifti1Image(img1, nii.affine)
            nii2 = nib.Nifti1Image(img2, nii.affine)
            if save:
                nib.save(nii1, path + prefix + '_ST' + str(i + 1) + str(j + 1) + '_ttest_fdr0.05.nii')
            #nib.save(nii2, path + prefix + '_ST' + str(i + 1) + str(j + 1) + '_ttest_fdr0.05_h.nii')
            Tmap.append(t)
            Pmap.append(p.copy())
    return np.array(Tmap), np.array(Pmap)

def cal_cluster_ANOVA_fdr(sheet,col_num,prefix,age,gender,onehotCenter,cluster,path,group,groupIndex,tiv,nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii",save=False):
    '''

    '''
    #L_BN = nib.load(r"C:\Users\DELL\OneDrive\brain\fsaverage_LR32k\fsaverage.L.BN_Atlas.32k_fs_LR.label.gii")
    #R_BN = nib.load(r"C:\Users\DELL\OneDrive\brain\fsaverage_LR32k\fsaverage.R.BN_Atlas.32k_fs_LR.label.gii")

    dscalar = nib.load(r"E:\matTool\workbench\HCP_WB_Tutorial_1.0\Q1-Q6_R440.MyelinMap_BC.32k_fs_LR.dscalar.nii")
    dlable = nib.load(r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii")

    BN_lable = np.asarray(dlable.dataobj)
    ciiData = np.zeros_like(BN_lable)

    
    # L_BN_surf = np.array(L_BN.agg_data())
    # R_BN_surf = np.array(R_BN.agg_data())
    #
    # dataArrayL = np.zeros_like(L_BN.agg_data())
    # dataArrayR = np.zeros_like(R_BN.agg_data())
    # newL = nib.gifti.gifti.GiftiImage(header=L_BN.header)
    # newR = nib.gifti.gifti.GiftiImage(header=R_BN.header)
    

    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    rois = np.unique(template[:])[1:]
    tiv = tiv.reshape([-1,1])
    f = np.zeros((col_num,))
    p = np.zeros((col_num,))
    img1 = np.zeros_like(template)
    img2 = np.zeros_like(template)

    for k, roi in zip(range(col_num), rois):
        var = np.array(sheet.col_values(k)[1:])
        
        var = regress_cov(var,np.concatenate([age,gender,onehotCenter,tiv],axis=1),center=False,keep_scale=False)
        var = var[group == groupIndex]
        f[k], p[k] = f_oneway(var[cluster == 0], var[cluster == 1], var[cluster == 2], var[cluster == 3])
    #print(p)
    _,p_fdr=fdrcorrection(p,0.05)
    #print(p_fdr)
    for k, roi in zip(range(col_num), rois):
        if p_fdr[k] <= 0.05:
            #print("pass")
            img1[template == roi] = f[k]
            img2[template == roi] = np.sign(f[k])
            
            ciiData[BN_lable == roi] = f[k]
            ciiData[(BN_lable == roi+210)] = f[k]
            #print(roi,2*roi-1,2*roi)
            # dataArrayL[L_BN_surf == roi] = f[k]
            # dataArrayR[R_BN_surf == roi] = f[k]
    nii1 = nib.Nifti1Image(img1, nii.affine)
    nii2 = nib.Nifti1Image(img2, nii.affine)
    
    # dataArrayL = nib.gifti.gifti.GiftiDataArray(dataArrayL)
    # dataArrayR = nib.gifti.gifti.GiftiDataArray(dataArrayR)
    #
    # newL.add_gifti_data_array(dataArrayL)
    # newR.add_gifti_data_array(dataArrayR)

    newCii = nib.cifti2.Cifti2Image(ciiData,dscalar.header)

    if(save):
        nib.save(nii1, path + prefix + '_ST' + '_anova_fdr0.05.nii')
        # newL.to_filename(path+prefix+'_ST_anova_fdr_L.gii')
        # newR.to_filename(path+prefix+'_ST_anova_fdr_R.gii')
        newCii.to_filename(path+prefix+'.dscalar.nii')
        if(col_num == 273):
            np.savetxt('MCAD_f.txt',f,fmt='%f')
    return p_fdr,f


def cal_cluster_T_fdr(sheet, col_num, prefix, age, gender, onehotCenter, cluster, path, group, groupIndex, tiv,
                          nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii", save=False):
    '''

    '''
    # L_BN = nib.load(r"C:\Users\DELL\OneDrive\brain\fsaverage_LR32k\fsaverage.L.BN_Atlas.32k_fs_LR.label.gii")
    # R_BN = nib.load(r"C:\Users\DELL\OneDrive\brain\fsaverage_LR32k\fsaverage.R.BN_Atlas.32k_fs_LR.label.gii")

    dscalar = nib.load(r"E:\matTool\workbench\HCP_WB_Tutorial_1.0\Q1-Q6_R440.MyelinMap_BC.32k_fs_LR.dscalar.nii")
    dlable = nib.load(
        r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii")

    BN_lable = np.asarray(dlable.dataobj)
    ciiData = np.zeros_like(BN_lable)

    # L_BN_surf = np.array(L_BN.agg_data())
    # R_BN_surf = np.array(R_BN.agg_data())
    #
    # dataArrayL = np.zeros_like(L_BN.agg_data())
    # dataArrayR = np.zeros_like(R_BN.agg_data())
    # newL = nib.gifti.gifti.GiftiImage(header=L_BN.header)
    # newR = nib.gifti.gifti.GiftiImage(header=R_BN.header)

    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    rois = np.unique(template[:])[1:]
    tiv = tiv.reshape([-1, 1])
    t = np.zeros((col_num,))
    p = np.zeros((col_num,))
    img1 = np.zeros_like(template)
    img2 = np.zeros_like(template)

    for k, roi in zip(range(col_num), rois):
        var = np.array(sheet.col_values(k)[1:])

        var = regress_cov(var, np.concatenate([age, gender, onehotCenter, tiv], axis=1), center=False, keep_scale=False)
        var = var[group == groupIndex]
        t[k], p[k] = stats.ttest_ind(var[cluster == 0], var[cluster == 1])
    _, p_fdr = fdrcorrection(p, 0.05)
    for k, roi in zip(range(col_num), rois):
        if p_fdr[k] <= 0.05:
            img1[template == roi] = t[k]
            ciiData[BN_lable == roi] = t[k]
            ciiData[(BN_lable == roi + 210)] = t[k]
    nii1 = nib.Nifti1Image(img1, nii.affine)
    newCii = nib.cifti2.Cifti2Image(ciiData, dscalar.header)
    if (save):
        nib.save(nii1, path + prefix + '_ST' + '_anova_fdr0.05.nii')
        # newL.to_filename(path+prefix+'_ST_anova_fdr_L.gii')
        # newR.to_filename(path+prefix+'_ST_anova_fdr_R.gii')
        newCii.to_filename(path + prefix + '.dscalar.nii')
        if (col_num == 273):
            np.savetxt('MCAD_f.txt', f, fmt='%f')
    return p_fdr, f
def cal_cluster_ttest(variable,cluster):
    result = np.zeros((6, 4))
    n=0
    for i in range(3):
        for j in range(i + 1, 4):
            result[n, 0] = i
            result[n, 1] = j
            result[n, 2], result[n, 3] = stats.ttest_ind(variable[cluster == i], variable[cluster == j])
            n = n + 1
    # result[:, 3] = cal_fdr(result[:, 3])
    return result

def cal_cluster_ttest_with_NC(sheet,col_num,prefix,age,gender,onehotCenter,cluster,group,path,groupIndex,tiv,nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii",save=False):
    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    Tmap = []
    Pmap = []
    rois = np.unique(template[:])[1:]
    tiv = tiv.reshape([-1,1])
    for i in range(4):
        t = np.zeros((col_num,))
        p = np.zeros((col_num,))
        img1 = np.zeros_like(template)
        img2 = np.zeros_like(template)
        for k, roi in zip(range(col_num), rois):
            var = np.array(sheet.col_values(k)[1:])
            var = regress_cov(var,np.concatenate([age,gender,onehotCenter,tiv],axis=1),center=False,keep_scale=False)
            ADData = var[(group == groupIndex)]
            ADData = ADData[cluster == i]
            NCData = var[group == 1]
            t[k], p[k] = stats.ttest_ind(ADData,NCData)
        _, p_fdr = fdrcorrection(p, 0.05)
        for k, roi in zip(range(col_num), rois):
            if (p_fdr[k]<0.05):
                #print("pass")
                img1[template == (roi)] = t[k]
                img2[template == (roi)] = np.sign(t[k])
        nii1 = nib.Nifti1Image(img1, nii.affine)
        nii2 = nib.Nifti1Image(img2, nii.affine)
        if save:
            nib.save(nii1, path + prefix + '_ST' + str(i + 1) + '_ttest_fdr0.05.nii')
        #nib.save(nii2, path + prefix + '_ST' + str(i + 1) + str(j + 1) + '_ttest_fdr0.05_h.nii')

        Tmap.append(t)
        Pmap.append(p.copy())
    return np.array(Tmap), np.array(Pmap)


def cal_cluster_ttest_with_other(sheet, col_num, prefix, age, gender, onehotCenter, cluster, group, path, groupIndex,tiv,
                              nii_path=r"E:\brain\subtype\subtype-data\BNA_2mm.nii", save=False):
    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    Tmap = []
    Pmap = []
    rois = np.unique(template[:])[1:]
    tiv = tiv.reshape([-1,1])
    for i in range(4):
        t = np.zeros((col_num,))
        p = np.zeros((col_num,))
        img1 = np.zeros_like(template)
        img2 = np.zeros_like(template)
        for k, roi in zip(range(col_num), rois):
            var = np.array(sheet.col_values(k)[1:])
            var = regress_cov(var, np.concatenate([age, gender, onehotCenter,tiv], axis=1), center=False, keep_scale=False)
            ADData = var[(group == groupIndex)]
            subtypeData = ADData[cluster == i]
            OtherData = ADData[cluster != i]
            t[k], p[k] = stats.ttest_ind(subtypeData, OtherData)
        _, p_fdr = fdrcorrection(p, 0.05)
        for k, roi in zip(range(col_num), rois):
            if (p_fdr[k] <= 0.05):
                # print("pass")
                img1[template == (roi)] = t[k]
                img2[template == (roi)] = np.sign(t[k])
        nii1 = nib.Nifti1Image(img1, nii.affine)
        nii2 = nib.Nifti1Image(img2, nii.affine)
        if save:
            nib.save(nii1, path + prefix + '_ST' + str(i + 1) + '_ttest_fdr0.05.nii')
        # nib.save(nii2, path + prefix + '_ST' + str(i + 1) + str(j + 1) + '_ttest_fdr0.05_h.nii')

        Tmap.append(t)
        Pmap.append(p.copy())
    return np.array(Tmap), np.array(Pmap)
def BrainAreaDisplay(Array):
    '''
    基本作用，传入273/210/246的脑区，返回对应脑区排序后的名称
    '''
    #Array = np.reshape(Array,[-1,])
    BN_atlas_info = pd.read_csv(r"E:\brain\BNatlas\brant_roi_info_274.csv")
    subNames = BN_atlas_info['label'].values
    lobeNames = BN_atlas_info['module'].values
    newNames = subNames +' '+ lobeNames
    indece = BN_atlas_info['index'].values

    sortedIndex = np.argsort(-Array)
    lobeNames = lobeNames[sortedIndex]
    subNames = subNames[sortedIndex]
    newNames = newNames[sortedIndex]
    return subNames.reshape([-1]),newNames.reshape([-1]),Array[sortedIndex].reshape([-1,1])

def cal_cluster_ANOVA_fdr_FHTM(sheet,col_num,prefix,age,gender,onehotCenter,cluster,path,group,groupIndex,tiv,nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii",save=False):
    L_BN = nib.load(r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.L.BN_Atlas.32k_fs_LR.label.gii")
    R_BN = nib.load(r"E:\brain\BNatlas\BN_Atlas_freesurfer\fsaverage\fsaverage_LR32k\fsaverage.R.BN_Atlas.32k_fs_LR.label.gii")

    L_BN_surf = np.array(L_BN.agg_data())
    R_BN_surf = np.array(R_BN.agg_data())

    dataArrayL = np.zeros_like(L_BN.agg_data())
    dataArrayR = np.zeros_like(R_BN.agg_data())
    newL = nib.GiftiImage(header=L_BN.header)
    newR = nib.GiftiImage(header=R_BN.header)

    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    rois = np.unique(template[:])[1:]

    VolumeData = np.zeros([age.shape[0],273])
    for i in range(273):
        VolumeData[:,i] = sheet.col_values(i)[1:]


    FHTM_volume = np.zeros([VolumeData.shape[0],6])
    FHTM_volume[:,0] = np.sum(VolumeData[:,14:28],axis=1)
    FHTM_volume[:,1] = np.sum(VolumeData[:,214:218],axis=1)
    FHTM_volume[:,2] = np.sum(VolumeData[:,68:80],axis=1)
    FHTM_volume[:,3] = np.sum(VolumeData[:,134:146],axis=1)
    FHTM_volume[:,4] = np.sum(VolumeData[:,188:210],axis=1)
    FHTM_volume[:,5] = np.sum(VolumeData[:,214:218],axis=1) / np.sum(VolumeData[:,0:210],axis=1)

    print('-----------------------------------FHTM---------------------------------')
    resData = pd.DataFrame(
        columns=['Item', 'F', 'P', 'ST1-2T', 'ST1-2P', 'ST1-3T', 'ST1-3P', 'ST1-4T', 'ST1-4P', 'ST2-3T', 'ST2-3P',
                 'ST2-4T', 'ST2-4P', 'ST3-4T', 'ST3-4P','NC_ST1_t','NC_ST1_p','NC_ST1_d','NC_ST2_t','NC_ST2_p','NC_ST2_d','NC_ST3_t','NC_ST3_p','NC_ST3_d','NC_ST4_t','NC_ST4_p','NC_ST4_d'])
    regFHTM_volume = pd.DataFrame()
    tRes = np.zeros([4,6,3])
    measure = ['Frontal','Hippocampus','Temporal','Parietal','Occipital','H:C']
    for k in range(6):
        dataDict = {}
        var = FHTM_volume[:,k]
        if(k < 5):
            var = regress_cov(var, np.concatenate([age, gender, onehotCenter,tiv.reshape([-1,1])], axis=1), center=False, keep_scale=False)

        for i in range(4):
            tvar = var[group == groupIndex]
            t,p = stats.ttest_ind(var[group==1],tvar[cluster==i])
            tRes[i,k,0] = t
            tRes[i,k,1] = p
            tRes[i,k,2] = (np.mean(var[group==1]) - np.mean(tvar[cluster==i]))/(np.sqrt((np.var(var[group==1])+np.var(tvar[cluster==i]))/2))

        f, p = f_oneway(tvar[cluster == 0], tvar[cluster == 1], tvar[cluster == 2], tvar[cluster == 3])
        print(f,p)

        tvar = (tvar - tvar.mean()) / tvar.std()
        result = cal_cluster_ttest(tvar, cluster)
        print(result)
        resData.loc[k] = [measure[k], f, p, result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                               result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                               result[5, 2], result[5, 3],tRes[0,k,0],tRes[0,k,1],tRes[0,k,2],tRes[1,k,0],tRes[1,k,1],tRes[1,k,2],tRes[2,k,0],tRes[2,k,1],tRes[2,k,2],tRes[3,k,0],tRes[3,k,1],tRes[3,k,2]]
        dataDict['Measure'] = [measure[k] for i in range(tvar.shape[0])]
        dataDict['value'] = tvar.reshape([-1,])
        dataDict['cluster'] = cluster.reshape([-1,])

        regFHTM_volume = regFHTM_volume.append(pd.DataFrame(dataDict))
    regFHTM_volume['cluster'] = regFHTM_volume['cluster'] + 1
    print('-----------------------------------FHTM---------------------------------')
    return regFHTM_volume,tRes,resData
def cal_cluster_ANOVA_fdr_SN_Volume(sheet,col_num,prefix,age,gender,onehotCenter,cluster,path,group,groupIndex,tiv,nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii",save=False):

    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    rois = np.unique(template[:])[1:]

    VolumeData = np.zeros([age.shape[0],273])
    for i in range(273):
        VolumeData[:,i] = sheet.col_values(i)[1:]

    FHTM_volume = np.zeros([VolumeData.shape[0],4])
    for i in range(4):
        idx = np.loadtxt('data/MCAD_Sub_SN' + str(i+1)+'_idx.txt').astype(np.int)
        FHTM_volume[:,i] = np.sum(VolumeData[:,idx],axis=1)


    print('-----------------------------------SN_Col---------------------------------')
    resData = pd.DataFrame(
        columns=['Item', 'F', 'P', 'ST1-2T', 'ST1-2P', 'ST1-3T', 'ST1-3P', 'ST1-4T', 'ST1-4P', 'ST2-3T', 'ST2-3P',
                 'ST2-4T', 'ST2-4P', 'ST3-4T', 'ST3-4P','NC_ST1_t','NC_ST1_p','NC_ST1_d','NC_ST2_t','NC_ST2_p','NC_ST2_d','NC_ST3_t','NC_ST3_p','NC_ST3_d','NC_ST4_t','NC_ST4_p','NC_ST4_d'])
    regFHTM_volume = pd.DataFrame()
    tRes = np.zeros([4,6,3])
    measure = ['SN1','SN2','SN3','SN4']
    for k in range(4):
        dataDict = {}
        var = FHTM_volume[:,k]
        var = regress_cov(var, np.concatenate([age, gender, onehotCenter,tiv.reshape([-1,1])], axis=1), center=False, keep_scale=False)
        tvar = var[group == groupIndex]
        for i in range(4):

            t,p = stats.ttest_ind(var[group==1],tvar[cluster==i])
            tRes[i,k,0] = t
            tRes[i,k,1] = p
            tRes[i,k,2] = (np.mean(var[group==1]) - np.mean(tvar[cluster==i]))/(np.sqrt((np.var(var[group==1])+np.var(tvar[cluster==i]))/2))

        f, p = f_oneway(tvar[cluster == 0], tvar[cluster == 1], tvar[cluster == 2], tvar[cluster == 3])
        print(f,p)

        #tvar = (tvar - tvar.mean()) / tvar.std()
        result = cal_cluster_ttest(tvar, cluster)
        print(result)
        resData.loc[k] = [measure[k], f[0], p[0], result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                               result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                               result[5, 2], result[5, 3],tRes[0,k,0],tRes[0,k,1],tRes[0,k,2],tRes[1,k,0],tRes[1,k,1],tRes[1,k,2],tRes[2,k,0],tRes[2,k,1],tRes[2,k,2],tRes[3,k,0],tRes[3,k,1],tRes[3,k,2]]


        dataDict['Measure'] = [measure[k] for i in range(tvar.shape[0]+var[group==1].shape[0])]
        dataDict['value'] = np.concatenate([tvar,var[group==1]]).reshape([-1,])
        dataDict['cluster'] = np.concatenate([cluster,[-1 for i in range(var[group==1].shape[0])]]).reshape([-1,])

        regFHTM_volume = regFHTM_volume.append(pd.DataFrame(dataDict))
    regFHTM_volume['cluster'] = regFHTM_volume['cluster'] + 1
    print('-----------------------------------SN_Vol---------------------------------')


    resData.to_csv('./data2/SubNetwork_ANOVA_Volume.csv')
    myco2 = sns.color_palette(["#79af97ff","#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF"])
    for key in measure:
        plt.figure(figsize=(3,3))
        sns.barplot(x='cluster', y='value',data = regFHTM_volume[regFHTM_volume['Measure'] == key],capsize=.1,errwidth=0.8,palette=myco2)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.savefig('./draw2/MCAD_Subnet%s_Volume_Bar.svg' % key,dpi=300,bbox_inches='tight')
        plt.show()


    return regFHTM_volume,tRes,resData
def getMCAD_DataMatrix(ALL_CONNECT,Group_1 = 1, Group_2 =3):
    '''
    获取功能连接数据，可以获得所有功能连接或216条数据,已被废弃
    param ALL_CONNECT: bool
    '''

    corrPath = setting.corrPath
    centerNames = setting.centerNames
    imageNames = setting.imageNames
    group = setting.group
    centers = setting.centers

    if ALL_CONNECT:
        INDEX = np.triu_indices_from(np.zeros([263, 263]), k=1)
    else:
        INDEX = tuple(altered_fc_idx.transpose())

    ADDataMatrix = np.zeros([INDEX[0].shape[0], imageNames[group == Group_2].shape[0]])
    NCDataMatrix = np.zeros([INDEX[0].shape[0], imageNames[group == Group_1].shape[0]])

    sampleP = 0
    for centerP, center in enumerate(centerNames):
        centerPath = os.path.join(corrPath, center)
        centerImageNames = imageNames[(centers == (centerP + 1)) & (group == Group_1)]
        for imageName in centerImageNames:
            filePath = os.path.join(centerPath, imageName + '_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
            NCDataMatrix[:, sampleP] = corrData[INDEX]
            sampleP += 1
    sampleP = 0
    for centerP, center in enumerate(centerNames):
        centerPath = os.path.join(corrPath, center)
        centerImageNames = imageNames[(centers == (centerP + 1)) & (group == Group_2)]
        for imageName in centerImageNames:
            filePath = os.path.join(centerPath, imageName + '_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
            ADDataMatrix[:, sampleP] = corrData[INDEX]
            sampleP += 1
    NCDataMatrix = 0.5 * (np.log((1 + NCDataMatrix) / (1 - NCDataMatrix)))
    ADDataMatrix = 0.5 * (np.log((1 + ADDataMatrix) / (1 - ADDataMatrix)))

    return NCDataMatrix,ADDataMatrix

def getADNI_DataMatrix(ALL_CONNECT = False,Group_1 = 1, Group_2 =3):
    Pathes = [r"E:\brain\subtype\subtype-data\ADNI\ADNI_ex\mean_ts"
        ,     r"D:\DATA\ADNI\ADNI_BNAtlas_mean_ts",
              r"H:\subtype-data\ADNI\FC\roi2roi_r_pearson_correlation"]
    tablePaths = ['./table3/NC_Define_Bl_All.csv',
                  './table/MCI_Define_Bl_with_cluster.csv',
                  './table/AD_Define_Bl_with_cluster.csv',
                  ]
    NC_Define_Bl = pd.read_csv(tablePaths[Group_1-1])
    AD_Define_Bl = pd.read_csv(tablePaths[Group_2-1])


    if ALL_CONNECT:
        INDEX = np.triu_indices_from(np.zeros([263, 263]), k=1)
    else:
        INDEX = tuple(altered_fc_idx.transpose())

    imageNames = NC_Define_Bl['Subject'].values
    dataFrom = NC_Define_Bl['DataFrom'].values
    NC_dataMatrix = np.zeros([INDEX[0].shape[0], imageNames.shape[0]])
    sampleP = 0
    for imageName, source in zip(imageNames, dataFrom):
        if (sampleP == 5):
            sampleP += 1
            continue
        if (source == 1):
            filePath = os.path.join(Pathes[0], imageName + '_ts.csv')
            corrData = calCorr(filePath)
        elif (source == 2):
            filePath = os.path.join(Pathes[1], imageName + '_ts.csv')
            corrData = calCorr(filePath)
        elif (source == 3):
            filePath = os.path.join(Pathes[2], imageName + '_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
        NC_dataMatrix[:, sampleP] = corrData[INDEX]
        sampleP += 1

    imageNames = AD_Define_Bl['Subject'].values
    dataFrom = AD_Define_Bl['DataFrom'].values
    AD_dataMatrix = np.zeros([INDEX[0].shape[0], imageNames.shape[0]])
    sampleP = 0
    for imageName, source in zip(imageNames, dataFrom):
        if (source == 1):
            filePath = os.path.join(Pathes[0], imageName + '_ts.csv')
            corrData = calCorr(filePath)
        elif (source == 2):
            filePath = os.path.join(Pathes[1], imageName + '_ts.csv')
            corrData = calCorr(filePath)
        elif (source == 3):
            filePath = os.path.join(Pathes[2], imageName + '_corr_r.txt')
            corrData = np.loadtxt(filePath, delimiter=',')
        AD_dataMatrix[:, sampleP] = corrData[INDEX]
        sampleP += 1
    NC_dataMatrix = 0.5 * (np.log((1 + NC_dataMatrix) / (1 - NC_dataMatrix)))
    AD_dataMatrix = 0.5 * (np.log((1 + AD_dataMatrix) / (1 - AD_dataMatrix)))
    return NC_dataMatrix, AD_dataMatrix



