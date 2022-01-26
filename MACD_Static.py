import numpy as np
import nibabel as nib
#import scipy.stats
from Core.subtypeUtils import regress_cov,cal_cluster_ANOVA_fdr,BrainAreaDisplay,cal_cluster_ANOVA_fdr_FHTM,cal_cluster_ttest_with_NC,cal_cluster_ttest,cal_cluster_ttest_map_posthoc,cal_cluster_ttest_with_other,to_categorical,cal_cluster_ANOVA_fdr_SN_Volume
from scipy.stats import f_oneway
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xlrd
import scipy
def showCurve(sheet ,col_num ,prefix ,age ,gender ,onehotCenter ,cluster ,path ,group ,groupIndex
              ,nii_path=r"E:\brain\BNatlas\BN_Atlas_274_with_cerebellum_without_255.nii" ,save=False):

    nii = nib.load(nii_path)
    template = np.nan_to_num(nii.get_fdata())
    rois = np.unique(template[:])[1:]

    disData = pd.DataFrame(columns=['data' ,'cluster' ,'area'])
    datas = []
    clusters = []
    areas = []

    for k, roi in zip(range(col_num), rois):
        var = np.array(sheet.col_values(k)[1:])

        var = regress_cov(var ,np.concatenate([age ,gender ,onehotCenter] ,axis=1) ,center=False ,keep_scale=False)
        var = var[group == groupIndex]

        var = var.flatten()
        var = var.tolist()
        datas.extend(var)
        clusters.extend(cluster.tolist())
        areas.extend([int(roi) for i in range(len(var))])

    disData['data'] = datas
    disData['cluster'] = clusters
    disData['area'] = areas

    plt.figure(figsize=[20 ,5])
    sns.lineplot(data=disData, x="area", y="data", hue="cluster" ,palette='muted')
    plt.savefig('./draw2/DIS.png' ,dpi=300)

def ANOVA_Volume_Cognitive(excelfile, cluster, groupIndex, onehotCenter, prefix=None, path='./',AD=True):
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
    tiv = np.array(sheet2.col_values(6)[1:])

    #_, nodalData = getGraphParameter('shortest_path_length', 3, thresh_type=0)

    # p_fdr_paramter,f = cal_paramter_ANOVA_fdr(nodalData,263,prefix+'Parameter',age.reshape([-1,1]),gender.reshape([-1,1]),onehotCenter,cluster,path,group,groupIndex=groupIndex,nii_path=r"E:\brain\BNatlas\BN_Atlas_6_centers.nii",save=True)
    # paramterIndex = np.where(p_fdr_paramter <=0.05)[0]
    # Tmap, Pmap = cal_paramter_ttest_map_posthoc(nodalData,paramterIndex,prefix+'Parameter',age.reshape([-1,1]),gender.reshape([-1,1]),onehotCenter,cluster=cluster,path=path,group=group,groupIndex=groupIndex,save=True)

    # ANOVA 分析子型间有差异的脑区
    fig, axes = plt.subplots(3, 4, figsize=[30, 40])
    p_fdr_gmv, f = cal_cluster_ANOVA_fdr(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]), gender.reshape([-1, 1]),
                                         onehotCenter, cluster, path, group,tiv=tiv,  groupIndex=groupIndex,save=True)
    subName, lobeName, qf = BrainAreaDisplay(-np.log(p_fdr_gmv))
    print(lobeName.shape, qf.shape)
    sns.heatmap(qf[:30], yticklabels=lobeName[:30], square=True, ax=axes[0, 0])

    showCurve(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]), gender.reshape([-1, 1]), onehotCenter, cluster, path,
              group, groupIndex=groupIndex, save=True)

    FHTM_volume_data, tRes, FHTM_resData = cal_cluster_ANOVA_fdr_FHTM(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]),
                                                        gender.reshape([-1, 1]), onehotCenter, cluster, path, group,tiv=tiv,
                                                        groupIndex=groupIndex, save=True)

    cal_cluster_ANOVA_fdr_SN_Volume(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]),
                                                        gender.reshape([-1, 1]), onehotCenter, cluster, path, group,tiv=tiv,
                                                        groupIndex=groupIndex, save=True)

    p_fdr_cc, f = cal_cluster_ANOVA_fdr(sheet4, 210, prefix + 'surf', age.reshape([-1, 1]), gender.reshape([-1, 1]),
                                        onehotCenter, cluster, path, group,tiv=tiv, groupIndex=groupIndex, save=True)
    subName, lobeName, qf = BrainAreaDisplay(-np.log(p_fdr_cc))
    sns.heatmap(qf[:30], yticklabels=lobeName[:30], square=True, ax=axes[0, 1])

    gmvIndex = np.where(p_fdr_gmv <= 0.05)[0]
    ccIndex = np.where(p_fdr_cc <= 0.05)[0]

    Tmap, Pmap = cal_cluster_ttest_map_posthoc(sheet3, gmvIndex, prefix + 'gmv', age.reshape([-1, 1]),
                                               gender.reshape([-1, 1]), onehotCenter, cluster=cluster, path=path,
                                               group=group, groupIndex=groupIndex, save=True)
    if(AD):
        if (np.sum(Pmap<0.05)>0):
            print("GMV STVSST MIN：",np.min(Tmap),np.max(Tmap[(Tmap<0) & (Pmap<0.05)]),'MAX',np.min(Tmap[(Tmap>0)& (Pmap<0.05)]),np.max(Tmap))
    Tmap2, Pmap2 = cal_cluster_ttest_map_posthoc(sheet4, ccIndex, prefix + 'surf', age.reshape([-1, 1]),
                                                 gender.reshape([-1, 1]), onehotCenter, cluster=cluster, path=path,
                                                 group=group, groupIndex=groupIndex, save=True)
    if(AD):
        if (np.sum(Pmap2 < 0.05) > 0):
            if((np.sum(Tmap2[(Tmap2<0) & (Pmap2<0.05)])>0) & (np.sum(Tmap2[(Tmap2>0) & (Pmap2<0.05)])>0)):
               print("SURF STVSST MIN：",np.min(Tmap2),np.max(Tmap2[(Tmap2<0)& (Pmap2<0.05)]),'MAX',np.min(Tmap2[(Tmap2>0)& (Pmap2<0.05)]),np.max(Tmap2))
            elif((np.sum(Tmap2[(Tmap2<0) & (Pmap2<0.05)])>0) | (np.sum(Tmap2[(Tmap2<0) & (Pmap2<0.05)])>0)):
               print("SURF STVSST MIN：",np.min(Tmap2),'MAX',np.max(Tmap2))

    Tmap, Pmap = cal_cluster_ttest_with_other(sheet3, 263, prefix + 'gmv_SO_', age.reshape([-1, 1]),
                                              gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                              groupIndex=groupIndex, path=path, group=group,tiv=tiv, save=True)
    Tmap, Pmap = cal_cluster_ttest_with_other(sheet4, 210, prefix + 'surf_SO_', age.reshape([-1, 1]),
                                              gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                              groupIndex=groupIndex, path=path, group=group,tiv=tiv, save=True)

    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet3, 263, prefix + 'gmv_S_', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                           groupIndex=groupIndex, path=path, group=group,tiv=tiv,save=True)
    Pmap = -np.log(Pmap)
    for i in range(4):
        T = Pmap[i]
        subName, lobeName, qTmap = BrainAreaDisplay(T)
        sns.heatmap(qTmap[:30], yticklabels=lobeName[:30], square=True, ax=axes[1, i])
    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet4, 210, prefix + 'surf_S_', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=cluster,tiv=tiv,
                                           groupIndex=groupIndex, path=path, group=group, save=True)
    Pmap = -np.log(Pmap)
    for i in range(4):
        T = Pmap[i]
        subName, lobeName, qTmap = BrainAreaDisplay(T)
        sns.heatmap(qTmap[:30], yticklabels=lobeName[:30], square=True, ax=axes[2, i])

    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet3, 263, prefix + 'gmv_ALL', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=np.ones(cluster.shape[0], ),tiv=tiv,
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
        if(pvalue<0.05):
            print(result)
        resData.loc[resRow] = [name, fvalue[0], pvalue[0], result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                               result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                               result[5, 2], result[5, 3]]
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
            if (pvalue < 0.05):
                print(result)

            resData.loc[resRow] = [name, fvalue[0], pvalue[0], result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                                   result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                                   result[5, 2], result[5, 3]]
            resRow += 1

    anovaData = pd.concat([anovaData, FHTM_volume_data])
    resData = pd.concat([resData, FHTM_resData])
    return anovaData, resData, tRes
def T_Volume_Cognitive(excelfile, cluster, groupIndex, onehotCenter, prefix=None, path='./',AD=True):
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
    tiv = np.array(sheet2.col_values(6)[1:])

    #_, nodalData = getGraphParameter('shortest_path_length', 3, thresh_type=0)

    # p_fdr_paramter,f = cal_paramter_ANOVA_fdr(nodalData,263,prefix+'Parameter',age.reshape([-1,1]),gender.reshape([-1,1]),onehotCenter,cluster,path,group,groupIndex=groupIndex,nii_path=r"E:\brain\BNatlas\BN_Atlas_6_centers.nii",save=True)
    # paramterIndex = np.where(p_fdr_paramter <=0.05)[0]
    # Tmap, Pmap = cal_paramter_ttest_map_posthoc(nodalData,paramterIndex,prefix+'Parameter',age.reshape([-1,1]),gender.reshape([-1,1]),onehotCenter,cluster=cluster,path=path,group=group,groupIndex=groupIndex,save=True)

    # ANOVA 分析子型间有差异的脑区
    fig, axes = plt.subplots(3, 4, figsize=[30, 40])
    p_fdr_gmv, f = cal_cluster_ANOVA_fdr(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]), gender.reshape([-1, 1]),
                                         onehotCenter, cluster, path, group,tiv=tiv,  groupIndex=groupIndex,save=True)
    subName, lobeName, qf = BrainAreaDisplay(-np.log(p_fdr_gmv))
    print(lobeName.shape, qf.shape)
    sns.heatmap(qf[:30], yticklabels=lobeName[:30], square=True, ax=axes[0, 0])

    showCurve(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]), gender.reshape([-1, 1]), onehotCenter, cluster, path,
              group, groupIndex=groupIndex, save=True)

    FHTM_volume_data, tRes, FHTM_resData = cal_cluster_ANOVA_fdr_FHTM(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]),
                                                        gender.reshape([-1, 1]), onehotCenter, cluster, path, group,tiv=tiv,
                                                        groupIndex=groupIndex, save=True)

    cal_cluster_ANOVA_fdr_SN_Volume(sheet3, 263, prefix + 'gmv', age.reshape([-1, 1]),
                                                        gender.reshape([-1, 1]), onehotCenter, cluster, path, group,tiv=tiv,
                                                        groupIndex=groupIndex, save=True)

    p_fdr_cc, f = cal_cluster_ANOVA_fdr(sheet4, 210, prefix + 'surf', age.reshape([-1, 1]), gender.reshape([-1, 1]),
                                        onehotCenter, cluster, path, group,tiv=tiv, groupIndex=groupIndex, save=True)
    subName, lobeName, qf = BrainAreaDisplay(-np.log(p_fdr_cc))
    sns.heatmap(qf[:30], yticklabels=lobeName[:30], square=True, ax=axes[0, 1])

    gmvIndex = np.where(p_fdr_gmv <= 0.05)[0]
    ccIndex = np.where(p_fdr_cc <= 0.05)[0]

    Tmap, Pmap = cal_cluster_ttest_map_posthoc(sheet3, gmvIndex, prefix + 'gmv', age.reshape([-1, 1]),
                                               gender.reshape([-1, 1]), onehotCenter, cluster=cluster, path=path,
                                               group=group, groupIndex=groupIndex, save=True)
    if(AD):
        if (np.sum(Pmap<0.05)>0):
            print("GMV STVSST MIN：",np.min(Tmap),np.max(Tmap[(Tmap<0) & (Pmap<0.05)]),'MAX',np.min(Tmap[(Tmap>0)& (Pmap<0.05)]),np.max(Tmap))
    Tmap2, Pmap2 = cal_cluster_ttest_map_posthoc(sheet4, ccIndex, prefix + 'surf', age.reshape([-1, 1]),
                                                 gender.reshape([-1, 1]), onehotCenter, cluster=cluster, path=path,
                                                 group=group, groupIndex=groupIndex, save=True)
    if(AD):
        if (np.sum(Pmap2 < 0.05) > 0):
            if((np.sum(Tmap2[(Tmap2<0) & (Pmap2<0.05)])>0) & (np.sum(Tmap2[(Tmap2>0) & (Pmap2<0.05)])>0)):
               print("SURF STVSST MIN：",np.min(Tmap2),np.max(Tmap2[(Tmap2<0)& (Pmap2<0.05)]),'MAX',np.min(Tmap2[(Tmap2>0)& (Pmap2<0.05)]),np.max(Tmap2))
            elif((np.sum(Tmap2[(Tmap2<0) & (Pmap2<0.05)])>0) | (np.sum(Tmap2[(Tmap2<0) & (Pmap2<0.05)])>0)):
               print("SURF STVSST MIN：",np.min(Tmap2),'MAX',np.max(Tmap2))

    Tmap, Pmap = cal_cluster_ttest_with_other(sheet3, 263, prefix + 'gmv_SO_', age.reshape([-1, 1]),
                                              gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                              groupIndex=groupIndex, path=path, group=group,tiv=tiv, save=True)
    Tmap, Pmap = cal_cluster_ttest_with_other(sheet4, 210, prefix + 'surf_SO_', age.reshape([-1, 1]),
                                              gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                              groupIndex=groupIndex, path=path, group=group,tiv=tiv, save=True)

    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet3, 263, prefix + 'gmv_S_', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=cluster,
                                           groupIndex=groupIndex, path=path, group=group,tiv=tiv,save=True)
    Pmap = -np.log(Pmap)
    for i in range(4):
        T = Pmap[i]
        subName, lobeName, qTmap = BrainAreaDisplay(T)
        sns.heatmap(qTmap[:30], yticklabels=lobeName[:30], square=True, ax=axes[1, i])
    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet4, 210, prefix + 'surf_S_', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=cluster,tiv=tiv,
                                           groupIndex=groupIndex, path=path, group=group, save=True)
    Pmap = -np.log(Pmap)
    for i in range(4):
        T = Pmap[i]
        subName, lobeName, qTmap = BrainAreaDisplay(T)
        sns.heatmap(qTmap[:30], yticklabels=lobeName[:30], square=True, ax=axes[2, i])

    Tmap, Pmap = cal_cluster_ttest_with_NC(sheet3, 263, prefix + 'gmv_ALL', age.reshape([-1, 1]),
                                           gender.reshape([-1, 1]), onehotCenter, cluster=np.ones(cluster.shape[0], ),tiv=tiv,
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
        if(pvalue<0.05):
            print(result)
        resData.loc[resRow] = [name, fvalue[0], pvalue[0], result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                               result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                               result[5, 2], result[5, 3]]
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
            if (pvalue < 0.05):
                print(result)

            resData.loc[resRow] = [name, fvalue[0], pvalue[0], result[0, 2], result[0, 3], result[1, 2], result[1, 3],
                                   result[2, 2], result[2, 3], result[3, 2], result[3, 3], result[4, 2], result[4, 3],
                                   result[5, 2], result[5, 3]]
            resRow += 1

    anovaData = pd.concat([anovaData, FHTM_volume_data])
    resData = pd.concat([resData, FHTM_resData])
    return anovaData, resData, tRes
def MCAD_AD_Statistic(cluster,savePath='./draw2/'):

    groupIndex = 3
    excelfile = xlrd.open_workbook('./mcad_info_687_del670.xlsx')
    center = np.asarray(excelfile.sheet_by_name('Sheet1').col_values(6)[1:])
    onehotCenter = to_categorical(center - 1, num_classes=7)
    ## anovaData 是归一化后的各个指标数据，可用于画图和导入R resData是方差分析和多重比较的结果 tRes是 FHTMOT和正常人的t检验结果，用于雷达图
    anovaData, resData, tRes = ANOVA_Volume_Cognitive(excelfile, cluster, groupIndex, onehotCenter, prefix='MCAD_AD_',
                                                      path=savePath)
    anovaData.to_csv('./data2/AD_anovaData.csv')
    resData.to_csv('./data2/AD_anovaRes.csv')
    return anovaData,resData,tRes

def MCAD_AD_Draw(anovaData,saveDir='./draw2/'):
    titles = ['TIV', 'relGMV', 'relWMV', 'relCSF', 'MMSE']
    anovaData['cluster'] = anovaData['cluster'].apply(lambda x: int(x))
    sns.set(font_scale=2, style='ticks')
    myco2 = sns.color_palette(["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF"])
    for column in ['relCSF', 'relGMV', 'relWMV', 'TIV', 'MMSE', 'AVLT', 'Dr', 'Ow', 'Nw', 'Age', 'Gender', 'Frontal',
                   'Hippocampus', 'Temporal', 'Parietal']:
        drawData = anovaData[anovaData['Measure'] == column]
        fig = plt.figure(figsize=[2, 2])
        ax = fig.add_subplot()
        g = sns.pointplot(y="value", x="cluster", data=drawData, col_wrap=9, dodge=True, ci=95, palette=myco2,
                          legend=False, capsize=0.1, margin_titles=False, ax=ax)
        if (column != 'Hippocampus'):
            g.set_title(column)
        else:
            g.set_title('Hippo')
        g.set_xlabel('')
        g.set_ylabel('')
        #g.set(ylim=(-1, 1))
        sns.despine(offset=2)

        plt.savefig(saveDir+'MCAD_AD_'+ column +'.svg',dpi=300,bbox_inches='tight')


def custom_boxplot(*args, **kwargs):

    from statannot import add_stat_annotation
    ax = sns.pointplot(*args, **kwargs)
    sns.despine(left=True)
    #plt.ylim([-0.8,0.8])
    compareDict = kwargs['compareDict']
    pairs = compareDict[kwargs['data']['Measure'].values[0]]
    if(pairs is not None):
            add_stat_annotation(
                ax, plot='barplot',
                data=kwargs['data'], x=kwargs['x'], y=kwargs['y'],
                comparisons_correction=None,order=['ST1','ST2','ST3','ST4'],
                test='t-test_ind', loc='outside', verbose=1, fontsize=10,
                box_pairs=pairs,
            )
def MCAD_AD_Draw_Bar(anovaData,compareDict,saveDir='./draw2/',columns = ['MMSE', 'AVLT','Age', 'Gender'],saveName='ADNI_MCI'):
    from statannot import add_stat_annotation


    titles = ['TIV', 'relGMV', 'relWMV', 'relCSF', 'MMSE']


    anovaData2 = anovaData.loc[anovaData['Measure'].isin(columns)]
    sns.set(font_scale=1)
    sns.set_theme(style="whitegrid")
    cmap = sns.color_palette(["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF"])

    print(anovaData2)

    g = sns.FacetGrid(data=anovaData2,col='Measure',col_order=columns,aspect=0.6)#,ylim=[-0.8,0.8])
    g.map_dataframe(custom_boxplot,x='cluster',y='value',palette=cmap,order=['ST1','ST2','ST3','ST4'],compareDict=compareDict)

    g.set_titles('')
    plt.savefig(saveDir + saveName + '.svg', dpi=300, bbox_inches='tight')
    # for i,column in enumerate(['relCSF', 'relGMV', 'relWMV', 'TIV', 'MMSE', 'AVLT', 'Dr', 'Ow', 'Nw', 'Age', 'Gender', 'Frontal',
    #                'Hippocampus', 'Temporal', 'Parietal']):
    #     drawData = anovaData[anovaData['Measure'] == column]
    #     plt.figure(figsize=[2, 2])
    #     ax = sns.pointplot(y="value", x="cluster",data=drawData, palette=cmap,order=['ST1','ST2','ST3','ST4'])
    #     box_pairs = [('ST1', 'ST2'),('ST1', 'ST3'),('ST1', 'ST4'),('ST2', 'ST3'),('ST2', 'ST4'),('ST3', 'ST4')]
    #     if(compare[i] is not None):
    #         add_stat_annotation(ax, plot='barplot', data=drawData, x='value', y='cluster',comparisons_correction=None,order=['ST1','ST2','ST3','ST4'],
    #                             box_pairs=compare[i], test='t-test_ind', loc='outside', verbose=1, fontsize=8)
    #     # plt.legend(loc='upper left', bbox_to_anchor=(1.03, 1))
    #     sns.despine(right=True,top=True)
    #     plt.tick_params(labelsize=10)
    #     plt.xlabel('', fontsize=14)
    #     plt.ylabel('', fontsize=14)
    #     plt.ylim([-1,1])
    #    plt.savefig(saveDir+'MCAD_AD_'+ column +'.png',dpi=300,bbox_inches='tight')


def MCAD_MCI_Statistic(savePath='./draw2/'):
    cluster = np.loadtxt('E:/brain/subtype/src/data/MCAD_MCI_Cluster_RC.txt')
    groupIndex = 2
    excelfile = xlrd.open_workbook('./mcad_info_687_del670.xlsx')
    center = np.asarray(excelfile.sheet_by_name('Sheet1').col_values(6)[1:])
    onehotCenter = to_categorical(center - 1, num_classes=7)
    MCIanovaData, MCIanovaRes, MCItRes = ANOVA_Volume_Cognitive(excelfile, cluster, groupIndex, onehotCenter,
                                                                prefix=None, path=savePath,AD=False)
    MCIanovaRes.to_csv('./data2/MCI_anovaRes.csv')
    MCIanovaData.to_csv('./data2/MCI_anovaData.csv')
    return MCIanovaData,MCIanovaRes,MCItRes

def MCAD_MCI_Draw(MCIanovaData,saveDir='./draw2/'):
    titles = ['TIV', 'relGMV', 'relWMV', 'relCSF', 'MMSE']
    MCIanovaData['cluster'] = MCIanovaData['cluster'].apply(lambda x: int(x))
    sns.set(font_scale=2, style='ticks')
    myco2 = sns.color_palette(["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF"])
    for column in ['relCSF', 'relGMV', 'relWMV', 'TIV', 'MMSE', 'AVLT', 'Dr', 'Ow', 'Nw', 'Age', 'Gender', 'Frontal',
                   'Hippocampus', 'Temporal', 'Parietal']:
        drawData = MCIanovaData[MCIanovaData['Measure'] == column]
        fig = plt.figure(figsize=[2, 2])
        ax = fig.add_subplot()
        g = sns.pointplot(y="value", x="cluster", data=drawData, col_wrap=9, dodge=True, ci=95, palette=myco2,
                          legend=False, capsize=0.1, margin_titles=False, ax=ax)
        if column != 'Hippocampus':
            g.set_title(column)
        else:
            g.set_title('Hippo')
        g.set_xlabel('')
        g.set_ylabel('')
        g.set(ylim=(-1, 1))
        sns.despine(offset=2)
        plt.savefig(saveDir+'MCAD_MCI_'+ column +'.svg',dpi=300,bbox_inches='tight')
def changeName(x):
    nameDict = {'AGE2': 'Age', 'PTGENDER2': 'Gender', 'RAVLT_learning': 'AVLT'}
    if (x in nameDict.keys()):
        return nameDict[x]
    else:
        return x
def DrawPointAndAnnotate(Data,columns,saveDir='./drawMCAD_Bar/',saveName='All_AD'):
    import sys
    sys.path.append('E:/brain/nits/')
    from stats import oneway_Anova
    compareDict = {}
    resData = pd.DataFrame(
        columns=['Item', 'F', 'P', 'ST1-2T', 'ST1-2P', 'ST1-3T', 'ST1-3P', 'ST1-4T', 'ST1-4P', 'ST2-3T', 'ST2-3P',
                 'ST2-4T', 'ST2-4P', 'ST3-4T', 'ST3-4P'])
    for c in columns:
        compareDict[c] = []
        resultsDict = {}
        res = oneway_Anova(Data[Data['Measure']==c],'value','cluster')
        resultsDict['Item'] = c
        resultsDict['F'] = res.loc['cluster','F']
        resultsDict['P'] = res.loc['cluster','PR(>F)']

        print('====================================================')
        print(res)
        paired_contrast = []
        for c1 in [1,2,3]:
            for c2 in range(c1+1,5):
                t,p = scipy.stats.ttest_ind(Data.loc[(Data['Measure']==c)&(Data['cluster']=='ST'+str(c1)),'value'],
                                            Data.loc[(Data['Measure']==c)&(Data['cluster']=='ST'+str(c2)),'value'])
                paired_contrast.append(['ST'+str(c1),'ST'+str(c2),t,p])
                if(p<0.05):
                    compareDict[c].append(('ST'+str(c1),'ST'+str(c2)))
                resultsDict['ST%d-%dT' % (c1,c2)] = t
                resultsDict['ST%d-%dP' % (c1, c2)] = p
        print(resData)
        resData = resData.append(pd.DataFrame(resultsDict,index=[0]))
        print(np.array(paired_contrast))
        print('====================================================')
    resData.to_csv('table/'+saveName+'.csv')
    for key,value in compareDict.items():
        if(len(value) == 0):
            compareDict[key] = None
    MCAD_AD_Draw_Bar(Data, saveDir=saveDir ,columns=columns,compareDict=compareDict,saveName=saveName)
if __name__ == '__main__':
    #cluster = np.loadtxt('./Cluster2/MCAD_AD_Cluster.txt')-1
    #anovaData, resData, tRes = MCAD_AD_Statistic(cluster,'./Cluster2/')
    #MCAD_AD_Draw(anovaData)

    import sys
    sys.path.append('E:/brain/nits/')
    from stats import oneway_Anova
    H_PATH = "data/MCAD_H_Combat.txt"
    W_PATH = "data/MCAD_W_Combat.txt"
    H = np.loadtxt(H_PATH)
    cluster = np.argmin(H, axis=0)
    anovaData, resData, tRes = MCAD_AD_Statistic(cluster)
    anovaData['cluster'] = anovaData['cluster'].apply(lambda x: 'ST'+str(int(x)))
    #MCAD_AD_Draw_Bar(anovaData,saveDir='./drawMCAD_Bar/',columns=['Age','Gender','MMSE','AVLT'],compareDict=compareDict)

    ADNI_anovaData = pd.read_csv('table2/ADNI_ANOVA_AD_MCI_ABETA.csv')
    ADNI_anovaData['Measure'] = ADNI_anovaData['item']
    ADNI_anovaData['cluster'] = ADNI_anovaData['cluster'].apply(lambda x: 'ST' + str(int(x)+1))
    ADNI_anovaData['Measure'] = ADNI_anovaData['Measure'].apply(changeName)
    columns = ['Age','Gender','MMSE', 'AVLT']
    for c in columns:
        ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'] = (ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'] - np.mean(ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'])) / np.std(ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'])
    All_anovaData = pd.concat([anovaData,ADNI_anovaData])
    DrawPointAndAnnotate(All_anovaData,columns,saveDir='./drawMCAD_Bar/',saveName='All_AD')

    columns = ['ADNI_EF', 'ADNI_MEM', 'ADNI_LAN', 'ADNI_VS','TAU','PTAU']
    for c in columns:
        ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'] = (ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'] - np.mean(ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'])) / np.std(ADNI_anovaData.loc[ADNI_anovaData['Measure']==c,'value'])
    DrawPointAndAnnotate(ADNI_anovaData, columns, saveDir='./drawMCAD_Bar/', saveName='ADNI_Composite')



    MCIanovaData, MCIanovaRes, MCItRes = MCAD_MCI_Statistic()
    MCIanovaData['cluster'] = MCIanovaData['cluster'].apply(lambda x: 'ST' + str(int(x)))
    ADNI_anovaData = pd.read_csv('table2/ADNI_ANOVA_Data_MCI_ABETA.csv')
    ADNI_anovaData['Measure'] = ADNI_anovaData['item']
    ADNI_anovaData['cluster'] = ADNI_anovaData['cluster'].apply(lambda x: 'ST' + str(int(x) + 1))
    ADNI_anovaData['Measure'] = ADNI_anovaData['Measure'].apply(changeName)
    columns = ['Age', 'Gender', 'MMSE', 'AVLT']
    for c in columns:
        ADNI_anovaData.loc[ADNI_anovaData['Measure'] == c, 'value'] = (ADNI_anovaData.loc[ADNI_anovaData[
                                                                                              'Measure'] == c, 'value'] - np.mean(
            ADNI_anovaData.loc[ADNI_anovaData['Measure'] == c, 'value'])) / np.std(
            ADNI_anovaData.loc[ADNI_anovaData['Measure'] == c, 'value'])
    All_anovaData = pd.concat([MCIanovaData, ADNI_anovaData])
    DrawPointAndAnnotate(All_anovaData, columns, saveDir='./drawMCAD_Bar/', saveName='MCI')
    columns = ['ADNI_EF', 'ADNI_MEM', 'ADNI_LAN', 'ADNI_VS']
    for c in columns:
        All_anovaData.loc[All_anovaData['Measure']==c,'value'] = (All_anovaData.loc[ADNI_anovaData['Measure']==c,'value'] - np.mean(All_anovaData.loc[All_anovaData['Measure']==c,'value'])) / np.std(All_anovaData.loc[All_anovaData['Measure']==c,'value'])
    DrawPointAndAnnotate(All_anovaData, columns, saveDir='./drawMCAD_Bar/', saveName='MCI_Composite')