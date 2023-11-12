#%%
import re

import numpy as np 
import pandas as pd

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
df_tmp = pd.read_excel('tg486_raw.xlsx')
df_tmp = df_tmp.dropna(subset = 'Genotoxicity').reset_index(drop = True)

# Genotoxicity  - negative/positive로 변환
response = ['negative', 'positive']
response_idx = [any(i in x for i in response) for x in df_tmp.Genotoxicity.unique()]
response_value = df_tmp.Genotoxicity.unique()[response_idx]

geno_tmp = df_tmp[df_tmp.Genotoxicity.isin(response_value)].reset_index(drop = True)
geno_tmp.Genotoxicity.unique()
geno_tmp.Genotoxicity.value_counts()


# CasRN
geno_tmp.CasRN.isna().sum()
(geno_tmp.CasRN == '-').sum()

casrn_drop_idx = geno_tmp.CasRN != '-'
geno = geno_tmp[casrn_drop_idx].reset_index(drop = True)


# multiiple results -> one result
# 하나의 화합물이 여러 개의 결과를 갖을 때, 보수적으로 진행하기 위해 하나라도 positive 결과를 갖으면 Genotoxicity를 postivie로 지정
def extract_conservative_endpoint(casrn):
    length = len(geno.Genotoxicity[geno.CasRN == casrn].unique())
    
    if length == 1:
        return geno.Genotoxicity[geno.CasRN == casrn].unique()[0]
    
    elif length > 1:
        # count = geno_tmp.Genotoxicity[geno_tmp.CasRN == casrn].value_counts()
        return 'positive'

geno_consv = geno[['Chemical', 'CasRN']].drop_duplicates(['CasRN']).reset_index(drop = True)
geno_consv['Genotoxicity'] = geno_consv.CasRN.map(lambda x: extract_conservative_endpoint(x))

geno_consv.Genotoxicity.value_counts()
geno_consv.Genotoxicity.value_counts(normalize = True)


# P개수 >= N개수면 P 
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
        else:
            return 'negative'


geno_maj = geno[['Chemical', 'CasRN']].drop_duplicates(['CasRN']).reset_index(drop = True)
geno_maj['Genotoxicity'] = geno_maj.CasRN.map(lambda x: extract_majority_endpoint(x))

geno_maj.Genotoxicity.value_counts()
geno_maj.Genotoxicity.value_counts(normalize = True)


#%%
geno_consv.to_excel('../tg486_consv.xlsx', index = False, header = True)
geno_maj.to_excel('../tg486_maj.xlsx', index = False, header = True)
