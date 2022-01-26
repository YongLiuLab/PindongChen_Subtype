import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multitest import fdrcorrection
import subtypeUtils
from husl import *
from .subtypeUtils import to_categorical,regress_cov,cal_cluster_ANOVA_fdr,cal_cluster_ttest,cal_cluster_ttest_map_posthoc,cal_cluster_ttest_with_NC,BrainAreaDisplay,cal_cluster_ANOVA_fdr_FHTM,getMCAD_DataMatrix,cal_cluster_ttest_with_other,cal_paramter_ANOVA_fdr,cal_paramter_ttest_map_posthoc


def cal_paramter_ANOVA_fdr(nodalData, col_num, prefix, age, gender, onehotCenter, cluster, path, group, groupIndex,
                           nii_path=r"E:\brain\BNatlas\BN_Atlas_6_centers.nii", save=False):
    '''

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


def getGraphParameter(parameter,thresh_ind,thresh_type=0):
    if(thresh_type == 0):
        prefix = 'Corr_'
    else:
        prefix = 'Sparse_'
    data = pd.read_csv('./data2/'+prefix+str(parameter)+'_thresh_'+str(thresh_ind)+'.csv',index_col=None).values
    globalData = data[:,0]
    nodalData = data[:,1:]
    return globalData,nodalData

def ANOVA_Volume_Cognitive(excelfile, cluster, groupIndex, onehotCenter, prefix=None, path='./'):
    '''
    MCAD mutiCenter gmv wmv avlt mmse ANOVA statistical analysis
    return statistical results to draw point figure
    '''
    if (prefix is None):
        if (groupIndex == 1):
            prefix = 'MCAD_NC_'
        elif (groupIndex == 2):
            prefix = 'MCAD_MCI_'
        elif (groupIndex == 3):
            prefix = 'MCAD_AD_'
    sheet = excelfile.sheet_by_name('Sheet1')
    group = np.array(sheet.col_values(2)[1:])

    gender = np.array(sheet.col_values(3)[1:])

    age = np.array(sheet.col_values(4)[1:])

    mmse = np.array(sheet.col_values(5)[1:])

    center = np.array(sheet.col_values(6)[1:])

    sheet2 = excelfile.sheet_by_name('tiv')
    sheet3 = excelfile.sheet_by_name('gmv')
    sheet4 = excelfile.sheet_by_name('surf')

    _, nodalData = getGraphParameter('shortest_path_length', 3, thresh_type=0)

    #p_fdr_paramter, f = cal_paramter_ANOVA_fdr(nodalData, 263, prefix + 'Parameter', age.reshape([-1, 1]),
    #                                           gender.reshape([-1, 1]), onehotCenter, cluster, path, group,
    #                                           groupIndex=groupIndex,
    #                                           nii_path=r"E:\brain\BNatlas\BN_Atlas_6_centers.nii", save=True)
    #paramterIndex = np.where(p_fdr_paramter <= 0.05)[0]
    #Tmap, Pmap = cal_paramter_ttest_map_posthoc(nodalData, paramterIndex, prefix + 'Parameter', age.reshape([-1, 1]),
    #                                            gender.reshape([-1, 1]), onehotCenter, cluster=cluster, path=path,
    #                                            group=group, groupIndex=groupIndex, save=True)

    # ANOVA 分析子型间有差异的脑区
    fig, axes = plt.subplots(3, 4, figsize=[30, 40])
    p_fdr_gmv, f = cal_cluster_ANOVA_fdr(sheet3, 273, prefix + 'gmv', age.reshape([-1, 1]), gender.reshape([-1, 1]),
                                         onehotCenter, cluster, path, group, groupIndex=groupIndex, save=True)
    subName, lobeName, qf = BrainAreaDisplay(-np.log(p_fdr_gmv))
    print(lobeName.shape, qf.shape)
    sns.heatmap(qf[:30], yticklabels=lobeName[:30], square=True, ax=axes[0, 0])

    FHTM_volume_data, tRes = cal_cluster_ANOVA_fdr_FHTM(sheet3, 273, prefix + 'gmv', age.reshape([-1, 1]),
                                                        gender.reshape([-1, 1]), onehotCenter, cluster, path, group,
                                                        groupIndex=groupIndex, save=True)

    p_fdr_cc, f = cal_cluster_ANOVA_fdr(sheet4, 210, prefix + 'surf', age.reshape([-1, 1]), gender.reshape([-1, 1]),
                                        onehotCenter, cluster, path, group, groupIndex=groupIndex, save=True)
    subName, lobeName, qf = BrainAreaDisplay(-np.log(p_fdr_cc))
    sns.heatmap(qf[:30], yticklabels=lobeName[:30], square=True, ax=axes[0, 1])

    gmvIndex = np.where(p_fdr_gmv <= 0.05)[0]
    ccIndex = np.where(p_fdr_cc <= 0.05)[0]

    Tmap, Pmap = cal_cluster_ttest_map_posthoc(sheet3, gmvIndex, prefix + 'gmv', age.reshape([-1, 1]),
                                               gender.reshape([-1, 1]), onehotCenter, cluster=cluster, path=path,
                                               group=group, groupIndex=groupIndex, save=True)
    Tmap2, Pmap2 = cal_cluster_ttest_map_posthoc(sheet4, ccIndex, prefix + 'surf', age.reshape([-1, 1]),
                                                 gender.reshape([-1, 1]), onehotCenter, cluster=cluster, path=path,
                                                 group=group, groupIndex=groupIndex, save=True)

    Tmap, Pmap = cal_cluster_ttest_with_other(sheet3, 273, prefix + 'gmv_SO_', age.reshape([-1, 1]),
                                              gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                              groupIndex=groupIndex, path=path, group=group, save=True)
    Tmap, Pmap = cal_cluster_ttest_with_other(sheet4, 210, prefix + 'surf_SO_', age.reshape([-1, 1]),
                                              gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                              groupIndex=groupIndex, path=path, group=group, save=True)

    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet3, 273, prefix + 'gmv_S_', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                           groupIndex=groupIndex, path=path, group=group, save=True)
    Pmap = -np.log(Pmap)
    for i in range(4):
        T = Pmap[i]
        subName, lobeName, qTmap = BrainAreaDisplay(T)
        sns.heatmap(qTmap[:30], yticklabels=lobeName[:30], square=True, ax=axes[1, i])
    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet4, 210, prefix + 'surf_S_', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                           groupIndex=groupIndex, path=path, group=group, save=True)
    Pmap = -np.log(Pmap)
    for i in range(4):
        T = Pmap[i]
        subName, lobeName, qTmap = BrainAreaDisplay(T)
        sns.heatmap(qTmap[:30], yticklabels=lobeName[:30], square=True, ax=axes[2, i])

    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet3, 273, prefix + 'gmv_ALL', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=np.ones(cluster.shape[0], ),
                                           groupIndex=groupIndex, path=path, group=group, save=True)

    gender = np.reshape(gender, [-1, 1])
    age = np.reshape(age, [-1, 1])

    rel_csf = np.array(sheet2.col_values(3)[1:])
    rel_gmv = np.array(sheet2.col_values(4)[1:])
    rel_wmv = np.array(sheet2.col_values(5)[1:])
    tiv = np.array(sheet2.col_values(6)[1:])
    nameDict = ['Basal ganglia', 'Prefrontal', 'Default-mode', 'Cingulate']
    anovaData = pd.DataFrame()

    resData = pd.DataFrame(
        columns=['Item', 'F', 'P', 'ST1-2T', 'ST1-2P', 'ST1-3T', 'ST1-3P', 'ST1-4T', 'ST1-4P', 'ST2-3T', 'ST2-3P',
                 'ST2-4T', 'ST2-4P', 'ST3-4T', 'ST3-4P'])
    resRow = 0
    for vars, name in zip([mmse, rel_csf, rel_gmv, rel_wmv, tiv, gender, age],
                          ['MMSE', 'relCSF', 'relGMV', 'relWMV', 'TIV', 'Gender', 'Age']):
        dataDict = {}
        if (name == 'Gender' or name == 'Age'):
            res = vars
        elif name != 'MMSE':
            res = regress_cov(vars,
                              np.concatenate((gender.reshape([-1, 1]), age.reshape([-1, 1]), onehotCenter), axis=1),
                              center=False, keep_scale=False)
        else:
            res = regress_cov(vars, np.concatenate((gender.reshape([-1, 1]), age.reshape([-1, 1])), axis=1),
                              center=False, keep_scale=False)
        res = res[group == groupIndex]
        res = (res - np.mean(res)) / np.std(res)

        dataDict['Measure'] = [name for i in range(res.shape[0])]
        dataDict['value'] = res.reshape([-1, ])
        dataDict['cluster'] = cluster.reshape([-1, ]) + 1
        anovaData = anovaData.append(pd.DataFrame(dataDict))

        fvalue, pvalue = f_oneway(res[cluster == 0], res[cluster == 1], res[cluster == 2], res[cluster == 3])
        print(name + 'ANOVA analysis F value: ' + str(fvalue) + ' P value: ' + str(pvalue))

        result = cal_cluster_ttest(res, cluster)
        print(result)
        resData.loc[resRow] = [name, fvalue[0], pvalue[0], result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                               result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                               result[4, 2], result[4, 3]]
        resRow += 1
    # 12列 13列 14列 15列分别是AVLT，延迟回忆，原词辨认和新词辨认
    items = ['AVLT', 'Dr', 'Ow', 'Nw', 'RC', 'MOCA', 'MOCA-1', 'MOCA-2', 'MOCA-3', 'MOCA-4', 'MOCA-5', 'MOCA-6',
             'MOCA-7', 'MMSE-1', 'MMSE-2', 'MMSE-3', 'MMSE-4', 'MMSE-5', 'MMSE-6', 'MMSE-7', 'MMSE-8', 'MMSE-9',
             'MMSE-10']
    cols = range(12, 12 + len(items))
    for col, name in zip(cols, items):
        dataDict = {}

        tempData = np.array(sheet.col_values(col)[1:])
        tempData2 = tempData[group == groupIndex]

        flag2 = np.where(tempData2 != '')
        flag = np.where(tempData != '')

        tgroup = group[flag]
        center_temp = center[flag]
        tempData = tempData[flag].astype(float)

        cluster_temp = cluster[flag2]
        tempData[center_temp != 5] = tempData[center_temp != 5] / 10
        tempData[center_temp == 5] = tempData[center_temp == 5] / 15

        res = regress_cov(tempData, np.concatenate((gender[flag].reshape([-1, 1]), age[flag].reshape([-1, 1])), axis=1),
                          center=False, keep_scale=False)
        res = res[tgroup == groupIndex]
        res = (res - np.mean(res)) / np.std(res)
        dataDict['Measure'] = [name for i in range(res.shape[0])]
        dataDict['value'] = res.reshape([-1, ])
        dataDict['cluster'] = cluster_temp.reshape([-1, ]) + 1
        anovaData = anovaData.append(pd.DataFrame(dataDict))

        if np.sum(cluster_temp == 0) * np.sum(cluster_temp == 1) * np.sum(cluster_temp == 2) * np.sum(
                cluster_temp == 3):
            print(np.sum(cluster_temp == 0), ',', np.sum(cluster_temp == 1), ',', np.sum(cluster_temp == 2), ',',
                  np.sum(cluster_temp == 3))
            fvalue, pvalue = f_oneway(res[cluster_temp == 0], res[cluster_temp == 1], res[cluster_temp == 2],
                                      res[cluster_temp == 3])
            print(name + ' ANOVA analysis F value: ' + str(fvalue) + ' P value: ' + str(pvalue))
            result = cal_cluster_ttest(res, cluster_temp)
            print(result)

            resData.loc[resRow] = [name, fvalue[0], pvalue[0], result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                                   result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                                   result[4, 2], result[4, 3]]
            resRow += 1

    anovaData = pd.concat([anovaData, FHTM_volume_data])
    return anovaData, resData, tRes

def getADNI_Survival_Table(ADNI_MERGE,VISTIME,MCI_Info,startWithBl=True):

    startTime = MCI_Info.loc[:,'Month_bl'].values

    ADNI_MCI_SURVIVAL_TIME = pd.DataFrame()
    sts = []
    change = []
    subjects = MCI_Info['PTID'].values.tolist()
    print(subjects)
    for i, subject in enumerate(subjects):
        resluts = ADNI_MERGE.loc[(ADNI_MERGE['PTID'] == subject)]
        time = resluts['Month_bl'].values.max()
        ST = resluts.iloc[0]['cluster']
        MCI_resluts = resluts[resluts['DX'] != 'Dementia']

        mciTime = MCI_resluts['Month_bl'].values.max()

        AD_results = resluts[resluts['DX'] == 'Dementia']

        if (AD_results.shape[0] != 0):
            status = 1
        else:
            status = 0

        sts.append(ST)
        change.append(status)
        if (mciTime == 0) and (time == 0):
            print('无随访：',subject)
            continue

        else:
            if(not startWithBl):
                mciTime = mciTime - startTime[i]
            if (mciTime < 0):
                print('随访时间不够：', subject)
                continue
            if (mciTime) > VISTIME:
                status = 0
                mciTime = VISTIME
            #print(subject, 'time:', time, ' mciTime', mciTime, '是否转化', status)

            ADNI_MCI_SURVIVAL_TIME = ADNI_MCI_SURVIVAL_TIME.append(
                pd.DataFrame({'PTID': subject, 'TIME': mciTime, 'STATUS': status, 'ST': ST}, index=[i]), ignore_index=True)
        #ADNI_MCI_SURVIVAL_TIME.to_csv(r"E:\brain\subtype\src/table\ADNI_MCI_Info_ST%s_TIME_110.csv" % str(VISTIME))
    return ADNI_MCI_SURVIVAL_TIME