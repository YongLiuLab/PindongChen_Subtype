from Core.subtypeUtils import washData,getADNI_TIV,regress_cov,columns,setting
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway,pearsonr
import numpy as np
def AD_Statitic():
    AD_Define_Bl = pd.read_csv('./table/AD_ABETA_cluster.csv')
    #AD_Define_Bl = pd.read_csv('./table/AD_Define_Bl_PYNMF.csv')
    CompositeTable = pd.read_csv(r"E:\brain\infoTable\ADNI\CompositeScore.csv")
    newTable_CTScore = pd.merge(AD_Define_Bl, CompositeTable, left_on=['RID', 'VISCODE3'], right_on=['RID', 'VISCODE2'],
                                how='left')
    newTable_CTScore.to_csv('E:/brain/subtype/src/table2/newTable_CTScore.csv')
    H = np.loadtxt('./data/ADNI_H_ABETA.txt')
    H = H[setting.ADNI_CLuster_Seq, :]
    H = H / np.sum(H,axis=0)

    AD_Define_Bl = newTable_CTScore
    dataDict = {}
    dataDict['cluster'] = []
    dataDict['value'] = []
    dataDict['item'] = []
    acolumns = columns.copy()
    acolumns.extend(['ADNI_EF', 'ADNI_MEM', 'ADNI_LAN', "ADNI_VS"])
    # acolumns.extend(['TIV','GMV','WMV','CSF'])

    numberDict = pd.DataFrame(columns=['item',1,2,3,4])

    titles = []
    for column in acolumns:

        staAD_Define_Bl = AD_Define_Bl[AD_Define_Bl[column].notnull()]
        staH = H[:,AD_Define_Bl[column].notnull()]
        if (staAD_Define_Bl[column][staAD_Define_Bl['cluster'] == 0].shape[0] *
                staAD_Define_Bl[column][staAD_Define_Bl['cluster'] == 1].shape[0] *
                staAD_Define_Bl[column][staAD_Define_Bl['cluster'] == 2].shape[0] *
                staAD_Define_Bl[column][staAD_Define_Bl['cluster'] == 3].shape[0] > 0):
            data = staAD_Define_Bl[column].values.reshape([-1, ])
            cl = staAD_Define_Bl['cluster'].values.reshape([-1, ])
            if (column == 'AGE2' or column == 'PTGENDER2'):
                data = data
            else:
                data = regress_cov(data,np.concatenate([staAD_Define_Bl.loc[:,'AGE2'].values.reshape([-1,1]),staAD_Define_Bl.loc[:,'PTGENDER2'].values.reshape([-1,1])],axis=1),center=False,keep_scale=False)
            #data = (data - np.mean(data)) / np.std(data)
            #data = data.reshape([-1,])
            f, p = f_oneway(data[cl == 0], data[cl == 1], data[cl == 2], data[cl == 3])
            print(column, f, p)


            corrDf = pd.DataFrame(columns=['value', 'loading', 'sub-network'])
            for h in range(4):
                r, p = pearsonr(staH[h, :], data.reshape([-1, ]))
                print(h + 1, ':', r, p)
                corrDf = corrDf.append(pd.DataFrame({'value': data.reshape([-1, ]), 'loading': staH[h, :],
                                                     'sub-network': ['sub-network%d' % (h + 1) for i in
                                                                     range(data.reshape([-1, ]).shape[0])]}))
            if (column in ['ADAS11', 'MMSE', 'ADNI_EF', 'ADNI_MEM', 'ADNI_LAN', 'ADNI_VS']):
                drawCor(corrDf, '_AD' + column)

            dataDict['cluster'].extend((cl.astype(np.int)).tolist())
            dataDict['value'].extend(data.reshape([-1, ]).tolist())
            dataDict['item'].extend([column for i in range(data.shape[0])])

            numberDict = numberDict.append(pd.DataFrame({'item':column,1:np.sum(cl==0),
                  2:staAD_Define_Bl[column][staAD_Define_Bl['cluster'] == 1].shape[0],
                  3:staAD_Define_Bl[column][staAD_Define_Bl['cluster'] == 2].shape[0],
                  4:staAD_Define_Bl[column][staAD_Define_Bl['cluster'] == 3].shape[0]},index=[0]))
            if (p < 0.05):
                #             plt.figure()
                #             g = sns.pointplot(data=dataDict,y=column,x='cluster',palette='muted')
                #             g.set(ylim=(-1,1))

                titles.append(column)

    dataDF = pd.DataFrame(dataDict)
    # AD_volumeData['cluster'] = AD_volumeData['cluster'].apply(lambda x : int(x))
    sns.set(font_scale=2, style='ticks')
    g = sns.catplot(y="value", x="cluster", col="item", data=dataDF, col_wrap=6, col_order=titles, kind="point",
                    dodge=True, height=3, aspect=1.25, palette='muted', scale=1.5, legend=False, margin_titles=False)
    g.set_titles('')
    g.set_xlabels('')
    g.set_ylabels('')
    for i, ax in enumerate(g.axes):
        ax.set(title=titles[i])
    # g.set(ylim=(-1,1))
    # g.savefig('AD_point.tif',dpi=300)
    plt.show()

    ADanovaData = pd.DataFrame(dataDict)
    # MCIanovaData['cluster'] = MCIanovaData['cluster'].apply(lambda x : int(x))
    ADanovaData.to_csv('./table2/ADNI_ANOVA_AD_MCI_ABETA.csv')
    numberDict.to_csv('./table/ADNI_AD_number_ABETA.csv')
def MCI_Statistic():
    MCI_Define_Bl = pd.read_csv('./table/MCI_ABETA_cluster.csv')
    #MCI_Define_Bl = pd.read_csv('./table/MCI_Define_Bl_PYNMF.csv')
    CompositeTable = pd.read_csv(r"E:\brain\infoTable\ADNI\CompositeScore.csv")
    newTable_CTScore_MCI = pd.merge(MCI_Define_Bl, CompositeTable, left_on=['RID', 'VISCODE3'],
                                    right_on=['RID', 'VISCODE2'], how='left')
    MCI_Define_Bl = newTable_CTScore_MCI
    ADNI_MCI_Cluster = MCI_Define_Bl['cluster'].values
    MCI_H = np.loadtxt('./data/ADNI_W_Combat_MCI_ABETA.txt')
    csum  = np.sum(MCI_H,axis=1)
    for i in range(4):
        MCI_H[:,i] = MCI_H[:,i] / csum
    dataDict_MCI = {}
    dataDict_MCI['cluster'] = []
    dataDict_MCI['value'] = []
    dataDict_MCI['item'] = []
    MCI_titles = []
    print(np.sum(ADNI_MCI_Cluster == 0), np.sum(ADNI_MCI_Cluster == 1), np.sum(ADNI_MCI_Cluster == 2),
          np.sum(ADNI_MCI_Cluster == 3))
    acolumns = columns.copy()
    acolumns.extend(['ADNI_EF', 'ADNI_MEM', 'ADNI_LAN', "ADNI_VS"])
    numberDict = pd.DataFrame(columns=['item',1, 2, 3, 4])
    for column in acolumns:
        if (column == 'TIV' or column == 'CSF' or column == 'WMV' or column == 'GMV' or column == 'AGE' or column == 'PTGENDER'):
            continue
        staMCI_Define_Bl = MCI_Define_Bl[MCI_Define_Bl[column].notnull()]
        staH = MCI_H[MCI_Define_Bl[column].notnull(),:]
        staMCI_Define_Bl = staMCI_Define_Bl[staMCI_Define_Bl['AGE2'].notnull()]
        staH = staH[staMCI_Define_Bl['AGE2'].notnull(), :]
        staMCI_Define_Bl = staMCI_Define_Bl[staMCI_Define_Bl['PTGENDER2'].notnull()]
        staH = staH[staMCI_Define_Bl['PTGENDER2'].notnull(), :]
        if (staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 0].shape[0] *
                staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 1].shape[0] *
                staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 2].shape[0] *
                staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 3].shape[0] < 1):
            continue
        numberDict = numberDict.append(
            pd.DataFrame({'item': column, 1: staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 0].shape[0],
                          2: staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 1].shape[0],
                          3: staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 2].shape[0],
                          4: staMCI_Define_Bl[column][staMCI_Define_Bl['cluster'] == 3].shape[0]},index=[0]))
        data = staMCI_Define_Bl[column].values.reshape([-1, ])
        if (data.shape[0] < 1):
            continue
        cl = staMCI_Define_Bl['cluster'].values.reshape([-1, ])
        if (column == 'AGE2' or column == 'PTGENDER2'):
            data = data
        else:
            data = regress_cov(data, np.concatenate([staMCI_Define_Bl.loc[:, 'AGE2'].values.reshape([-1, 1]),
                                                 staMCI_Define_Bl.loc[:, 'PTGENDER2'].values.reshape([-1, 1])], axis=1),
                           center=False, keep_scale=False)
        #data = (data - np.mean(data)) / np.std(data)
        #data = data.reshape([-1,])
        f, p = f_oneway(data[cl == 0], data[cl == 1], data[cl == 2], data[cl == 3])
        print(column, f, p)

        corrDf = pd.DataFrame(columns=['value','loading','sub-network'])
        for h in range(4):
            r, p = pearsonr(staH[:, h], data.reshape([-1, ]))
            print(h + 1, ':', r, p)
            data = data.astype(np.float)
            corrDf = corrDf.append(pd.DataFrame({'value':data.reshape([-1,]),'loading':staH[:,h],'sub-network':['sub-network%d' %(h+1) for i in range(staH.shape[0])]}))
        if(column in ['ADAS11','MMSE','ADNI_EF','ADNI_MEM','ADNI_LAN','ADNI_VS']):
            drawCor(corrDf,'_MCI'+column)
        dataDict_MCI['cluster'].extend((cl.astype(np.int)).tolist())
        dataDict_MCI['value'].extend(data.reshape([-1, ]).tolist())
        dataDict_MCI['item'].extend([column for i in range(data.shape[0])])
        if (p < 0.05):
            MCI_titles.append(column)

    dataDF = pd.DataFrame(dataDict_MCI)
    sns.set(font_scale=2, style='ticks')
    g = sns.catplot(y="value", x="cluster", col="item", data=dataDF, col_wrap=5, col_order=MCI_titles, kind="point",
                    dodge=True, height=3, aspect=1.25, palette='muted', scale=1.5, legend=False, margin_titles=False)
    g.set_titles('')
    g.set_xlabels('')
    g.set_ylabels('')
    for i, ax in enumerate(g.axes):
        ax.set(title=MCI_titles[i])
    g.set(ylim=(-1, 1))
    plt.show()
    # g.savefig('AD_point.tif',dpi=300)

    MCIanovaData = pd.DataFrame(dataDict_MCI)
    MCIanovaData.to_csv('./table2/ADNI_ANOVA_Data_MCI_ABETA.csv')
    numberDict.to_csv('./table/ADNI_MCI_number_ABETA.csv')

def drawCor(dF,prefix):
    from scipy import stats
    colors = ["#374E55FF", "#DF8F44FF", "#00A1D5FF", "#B24745FF"]
    cmap = sns.color_palette(colors)

    plt.figure(figsize=(3, 3))
    #g = sns.regplot(ADNI_t, MCAD_t,line_kws={'color':colors[i]},scatter_kws={'color':'black','s':4})
    g = sns.lmplot(data=dF,x='loading',y='value',palette=cmap,hue='sub-network')

    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    plt.savefig('draw2/SubnetworkCorrClinical%s.svg' % prefix, dpi=300, bbox_inches='tight')
def SN_Cor():
    from Core.dataload import getADNI_X_Norm
    from scipy.stats import pearsonr
    from Core.subtypeUtils import columns
    ADNI_NC, ADNI_Patient = getADNI_X_Norm(Group_id=1, is_Origin=True), getADNI_X_Norm(Group_id=3, is_Origin=True)

    ADNI_NC = 0.5 * (np.log((1 + ADNI_NC) / (1 - ADNI_NC)))
    ADNI_Patient = 0.5 * (np.log((1 + ADNI_Patient) / (1 - ADNI_Patient)))
    ADNI_NC = np.nan_to_num(ADNI_NC)
    ADNI_Patient = np.nan_to_num(ADNI_Patient)

    ADNI_cluster = pd.read_csv('./table/AD_ABETA_cluster.csv')['cluster'].values
    NC_Info = pd.read_csv('table/NC_ABETA.csv')
    AD_Info = pd.read_csv('table/AD_ABETA_cluster.csv')
    CompositeTable = pd.read_csv(r"E:\brain\infoTable\ADNI\CompositeScore.csv")
    newTable_CTScore = pd.merge(AD_Info, CompositeTable, left_on=['RID', 'VISCODE3'], right_on=['RID', 'VISCODE2'],
                                how='left')

    ADNI_AD_Age = AD_Info['AGE2'].values
    ADNI_NC_Age = NC_Info['AGE2'].values
    ADNI_AD_Gender = AD_Info['PTGENDER2'].values
    ADNI_NC_Gender = NC_Info['PTGENDER2'].values


    SN_Index = []

    SN_NC_Data = np.zeros([NC_Info.shape[0],4])
    SN_AD_Data = np.zeros([AD_Info.shape[0], 4])
    for i in range(4):
        SN_INDEX = np.loadtxt('data/SN_%d_INDEX.txt' % (i+1)).astype(np.int)
        SN_NC_Data[:,i] = np.mean(ADNI_NC[SN_INDEX, :],axis=0)
        SN_AD_Data[:,i] = np.mean(ADNI_Patient[SN_INDEX, :], axis=0)
    acolumns = columns.copy()
    acolumns.extend(['ADNI_EF', 'ADNI_MEM', 'ADNI_LAN', "ADNI_VS"])
    for name in acolumns:
        print(name)
        for i in range(4):

            score = newTable_CTScore[name].values
            index = ~np.isnan(score)
            if(sum(index)>20):
                r,p = pearsonr(SN_AD_Data[index,i],score[index])
                if(p<0.05):
                    print('ST%d:'%(i+1),r,p)

if __name__ == '__main__':
    AD_Statitic()
    MCI_Statistic()
