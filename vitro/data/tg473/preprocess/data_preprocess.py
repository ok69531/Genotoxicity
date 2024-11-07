#%%
import re

import numpy as np 
import pandas as pd

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
df_tmp = pd.read_excel('tg473_raw.xlsx')
df_tmp = df_tmp.dropna(subset = 'Genotoxicity').reset_index(drop = True)

# pn_map = pd.DataFrame({'raw': df_tmp.Genotoxicity.unique()})
# pn_map['genotoxicity'] = np.nan
# pn_map['genotoxicity'] = list(map(lambda x: other_map.get(x), pn_map.raw))
# pn_map['genotoxicity'][22] = 'positive'
# pn_map.to_csv('tg473_pn_map.csv', header = True, index = False)

# Genotoxicity  - negative/positive로 변환
response = ['negative', 'positive']
response_idx = [any(i in x for i in response) for x in df_tmp.Genotoxicity.unique()]
response_value_tmp = df_tmp.Genotoxicity.unique()[response_idx]
other_drop_value = ['other: positive structural, negative numerical', 
                    'other: The positive result was considered to  be due to a cytotoxic mechanism and to have no biological relevance',
                    'other: The finding in this study is very likely to be a false positive due to osmotic effects',
                    'other: Significant chromosome damage at 28 µg/mL; negative at 7 and 14 µg/mL.',
                    'other: no clear dose related positive response',
                    'other: No recordable chromosome aberations relative to positive, vehicle and historical controls.',
                    'other: positive for the induction of structural and negative for the induction of numerical chromosome aberrations',
                    'other: false positive results due to a decrease in the pH value',
                    'other: questionable positive',
                    'other: considered positive structural, negative numerical',
                    'other: without metabolic activation: negative; with metabolic activation: positive at 252 and 378 µg/ml (20h treatment period)',
                    'other: ambiguous results in 20+0 h treatment in the absence of S-9, whereas, negative results in 3+17 h treatments in both the absence and presence of S-9',
                    'other: Negative for the induction of structural chromosomal aberrations and positive for the induction of numerical chromosomal aberrations',
                    'other: False positive due to cytotoxicity']
response_value = [x for x in response_value_tmp if x not in other_drop_value]

geno_tmp = df_tmp[df_tmp.Genotoxicity.isin(response_value)].reset_index(drop = True)
geno_tmp.Genotoxicity.unique()
geno_tmp.Genotoxicity.value_counts()

other_idx = [i for i in range(len(geno_tmp)) if 'other:' in geno_tmp.Genotoxicity[i]]
other_map = {'other: weakly positive': 'positive',
             'other: positive, but lacking in dose-response (LBI) / only one dose positive (CU), trend P<0.005, for SCE induction; confirmation required': 'positive',
             'other: negative (LBI) / only one dose positive (CU, contributed by simple breaks), for CA induction': 'negative',
             'other: positive responses were noted at overtly toxic concentrations: +S9: at 20 µg/mL and -S9: at 156 µg/mL': 'positive',
             'other: clastogenicity negative': 'negative',
             'other: polyploidy positive at precipitating and cytotoxic concentrations.': 'positive',
             'other: Weakly positive': 'positive',
             'other: Clastogenicity: positive': 'positive'}
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
geno_consv.to_excel('../tg473_consv.xlsx', index = False, header = True)
geno_maj.to_excel('../tg473_maj.xlsx', index = False, header = True)
