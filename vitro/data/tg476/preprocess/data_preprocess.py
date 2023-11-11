#%%
import re

import numpy as np 
import pandas as pd

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
df_tmp = pd.read_excel('tg476_raw.xlsx')
df_tmp = df_tmp.dropna(subset = 'Genotoxicity').reset_index(drop = True)

# Genotoxicity  - negative/positive로 변환
response = ['negative', 'positive']
response_idx = [any(i in x for i in response) for x in df_tmp.Genotoxicity.unique()]
response_value_tmp = df_tmp.Genotoxicity.unique()[response_idx]
other_drop_value = ['other: negative for trifluorothymidine resistance with S9; inconclusive without S9',
                    'other: questionable (1st trial) & positive (2nd trial)',
                    'other: not evaluated: because of inadequate response by the positive and negative controls, data from this experiment were considered meaningless',
                    'other: False positive',
                    'other: (-S9) = equivocal (defined in the study report as the absence of a dose-response relationship, but the presence of a two-fold increase in mutant frequency over background at any of the doses examined) and (+S9) = negative',
                    'other: weakly positive (acc. to evaluation criteria layed down in the study report); ambigious (acc. to lately recommended evaluation criteria)',
                    'other: positive with S9 mix, negative in coculture with rat hepatocytes']
response_value = [x for x in response_value_tmp if x not in other_drop_value]

geno_tmp = df_tmp[df_tmp.Genotoxicity.isin(response_value)].reset_index(drop = True)
geno_tmp.Genotoxicity.unique()
geno_tmp.Genotoxicity.value_counts()

other_idx = [i for i in range(len(geno_tmp)) if 'other:' in geno_tmp.Genotoxicity[i]]
other_map = {'other: experiment was repeated for clarification and found negative': 'negative',
             'other: weak positive': 'positive',
             'other: Weakly positive , a weak increase in the induction of gene mutations was observed at the highest investigated concentration (1600 µg/mL) in the first experiment.': 'positive',
             'other: positive only in highly cytotxic concentrations': 'positive',
             'other: weakly positive': 'positive',
             'other: positive only with cytotoxicity in the absence of S9': 'positive'}
geno_tmp.Genotoxicity[other_idx] = list(map(lambda x: other_map.get(x), geno_tmp.Genotoxicity[other_idx]))


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
geno_consv.to_excel('../tg476_consv.xlsx', index = False, header = True)
geno_maj.to_excel('../tg476_maj.xlsx', index = False, header = True)
