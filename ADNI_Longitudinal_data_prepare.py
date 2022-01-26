import pandas as pd

import numpy as np
AD_Long = pd.read_csv('./table/AD_Cog_Long_ABETA.csv')

ADNI_MERGE = pd.read_csv('E:/brain/infoTable/ADNI/ADNIMERGE_last.csv',low_memory=False)
ComScore = pd.read_csv('E:/brain/infoTable/ADNI/CompositeScore.csv')
AD_Info = pd.read_csv('./table/AD_Define_Bl_with_cluster_ABETA.csv')

ADNI_MERGE = pd.merge(ADNI_MERGE,ComScore,left_on=['RID','VISCODE'],right_on=['RID','VISCODE2'])
AD_Info = pd.merge(AD_Info,ComScore,left_on=['RID','VISCODE3'],right_on=['RID','VISCODE2'])
AD_Info.loc[:,'VIS_M'] = AD_Info['VISCODE3'].apply(lambda x : 0 if x=='bl' else int(x[1:]))
ADNI_MERGE.loc[:,'VIS_M'] = ADNI_MERGE['VISCODE_x'].apply(lambda x : 0 if x=='bl' else int(x[1:]))

Long_Info = pd.DataFrame(columns=["PTID",'VIS_M','Item','cluster','Change','Age','Gender'])

isBL = True
VIS_RANGE = 121


for index, row in AD_Info.iterrows():
    print((row['PTID']),'VISCODE',row['VISCODE3'])
    LongInfo = ADNI_MERGE.loc[ADNI_MERGE.PTID == row['PTID']]
    if not isBL:
        LongInfo = LongInfo.loc[LongInfo.VIS_M >= row['VIS_M']]
    for column in columns:
        cLongInfo = LongInfo.loc[pd.notna(LongInfo[column])]
        cLongInfo = cLongInfo.loc[(cLongInfo['DX'] == 'Dementia') | (cLongInfo['DX'] == 'AD')]
        if(cLongInfo.shape[0]<1):
            continue
        if not isBL:
            bl = row.loc[column]
            VIS_bl = row.loc['VIS_M']
        else:
            VIS_bl = cLongInfo['VIS_M'].values.min()
            bl = cLongInfo.loc[cLongInfo['VIS_M']==VIS_bl,column].values
        cLongInfo.loc[:,'Change'] = cLongInfo.loc[:,column] - bl
        cLongInfo.loc[:,'VIS_M'] = cLongInfo.loc[:,'VIS_M'] - VIS_bl
        cLongInfo.loc[:,'Item'] = [column for i in range(cLongInfo.shape[0])]
        cLongInfo.loc[:,'cluster'] = [row['cluster'] for i in range(cLongInfo.shape[0])]
        cLongInfo.loc[:, 'Age'] = [row['AGE2'] for i in range(cLongInfo.shape[0])]
        cLongInfo.loc[:,'Gender'] = [row['PTGENDER2'] for i in range(cLongInfo.shape[0])]
        Long_Info = Long_Info.append(cLongInfo.loc[:,["PTID",'VIS_M','Item','cluster','Change','Age','Gender']])

Long_Info.to_csv('./table/AD_Cog_Long_ABETA.csv')


import numpy as np
AD_Long = pd.read_csv('./table/AD_Cog_Long_ABETA.csv')
for key in ['MMSE','ADAS11','ADNI_EF','ADNI_LAN','ADNI_VS','ADNI_MEM']:
    tAD_Long = AD_Long[AD_Long['Item']==key]
    print(key)
    for c in range(4):
        for i in [12,24,36,48,60,72,84,96,108,120]:
            print(np.unique(tAD_Long.loc[(tAD_Long['VIS_M']>=i) & (tAD_Long['cluster'] == c),'PTID']).shape[0],end=',')
        print()
    print()

