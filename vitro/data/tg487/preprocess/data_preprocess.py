#%%
import re

import numpy as np 
import pandas as pd

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
df_tmp = pd.read_excel('tg487_raw.xlsx')
df_tmp = df_tmp.dropna(subset = 'Genotoxicity').reset_index(drop = True)

pn_map = pd.read_csv('tg487_pn_map.csv')
pn_map = pn_map.set_index('raw').to_dict()['genotoxicity']


#%%
# pn_map = pd.DataFrame({'raw': df_tmp.Genotoxicity.unique()})
# pn_map['genotoxicity'] = np.nan
# pn_map['genotoxicity'][2] = 'positive'
# pn_map.to_csv('tg487_pn_map.csv', header = True, index = False)


#%%
df_tmp.Genotoxicity = df_tmp.Genotoxicity.map(lambda x: pn_map[x])
geno_tmp = df_tmp[df_tmp.Genotoxicity.notna()].reset_index(drop = True)

geno_tmp.Genotoxicity.unique()
geno_tmp.Genotoxicity.value_counts()


# CasRN
geno_tmp.CasRN.isna().sum()
(geno_tmp.CasRN == '-').sum()

casrn_drop_idx = geno_tmp.CasRN != '-'
geno = geno_tmp[casrn_drop_idx].reset_index(drop = True)
result = geno[['Chemical', 'CasRN']].drop_duplicates(['CasRN']).reset_index(drop = True)


# multiiple results -> one result
# 하나의 화합물이 여러 개의 결과를 갖을 때, 보수적으로 진행하기 위해 하나라도 positive 결과를 갖으면 Genotoxicity를 postivie로 지정
def extract_conservative_endpoint(casrn):
    length = len(geno.Genotoxicity[geno.CasRN == casrn].unique())
    
    if length == 1:
        return geno.Genotoxicity[geno.CasRN == casrn].unique()[0]
    
    elif length > 1:
        # count = geno_tmp.Genotoxicity[geno_tmp.CasRN == casrn].value_counts()
        return 'positive'

result['consv'] = result.CasRN.map(lambda x: extract_conservative_endpoint(x))

result.consv.value_counts()
result.consv.value_counts(normalize = True)


# P개수 > N개수면 P 
def extract_majority_endpoint(casrn):
    uniq_val = geno.Genotoxicity[geno.CasRN == casrn].unique()
    length = len(uniq_val)
    
    if length == 1:
        return uniq_val[0]
    
    elif length > 1:
        val_count = geno.Genotoxicity[geno.CasRN == casrn].value_counts()
        
        num_neg = val_count['negative']
        num_pos = val_count['positive']
        
        if num_pos >= num_neg:
            return 'positive'
        elif num_pos == num_neg:
            return np.nan
        else:
            return 'negative'


result['maj'] = result.CasRN.map(lambda x: extract_majority_endpoint(x))
result.maj.isna().sum()
result.maj.notna().sum()

result.maj.value_counts()
result.maj.value_counts(normalize = True)


#%%
result.to_excel('../tg487_tmp.xlsx', index = False, header = True)
