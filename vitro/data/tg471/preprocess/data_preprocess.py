#%%
import re

import numpy as np 
import pandas as pd

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)


#%%
df_tmp = pd.read_excel('tg471_raw.xlsx')
df_tmp = df_tmp.dropna(subset = 'Genotoxicity').reset_index(drop = True)

# Genotoxicity  - negative/positive로 변환
response = ['negative', 'positive']
response_idx = [any(i in x for i in response) for x in df_tmp.Genotoxicity.unique()]
response_value_tmp = df_tmp.Genotoxicity.unique()[response_idx]
other_drop_value = ['other: No positive response observed', 
                    'other: result could not be used for judgement because of failure of positive control']
response_value = [x for x in response_value_tmp if x not in other_drop_value]

geno_tmp = df_tmp[df_tmp.Genotoxicity.isin(response_value)].reset_index(drop = True)
geno_tmp.Genotoxicity.unique()
geno_tmp.Genotoxicity.value_counts()

other_idx = [i for i in range(len(geno_tmp)) if 'other:' in geno_tmp.Genotoxicity[i]]
other_map = {
    'other: positive in strains TA 1537 and TA 98 in the absence of metabolic activation': 'positive',
    'other: positive but markedly reduced with respect to TA98': 'positive',
    'other: positive, but markedly reduced compared to TA100': 'positive',
    'other: weak positive': 'positive',
    'other: negative without S9 mix, questionable/ambiguous with S9 mix': 'negative',
    'other: ambiguous in the first trial, negative in the second trial': 'negative',
    'other: weakly positive when reevaluated by SCF negative': 'positive',
    "other: negative (a slight increase in TA 1536 and TA 98 in one test could not be confirmed in two additional tests, which clearly didn't show any increase in the number of revertant colonies)": 'negative',
    'other: without S9:ambiguous; with S9: negative': 'negative',
    'other: negative at 20, 80, 320 and 1280 μg, positive at 5120 ug': 'positive',
    'other: weak positive effect at far in excess concentrations (>= 160 mg/plate)': 'positive',
    'other: Equivocal response in Salmonella strain TA100, both with and without microsomal activation, but was clearly negative in E. coli strain WP2uvrA and Salmonella strains TA98, TA1535 and TA1537.': 'negative',
    'other: weak mutagen only without S9; with S9 negative': 'negative',
    'other: positive only in Salmonella typhimurium TA 100': 'positive',
    'other: S typhimurium TA1537 and TA 98 gave positive results: TA1535 and TA 100 yielded negative results': 'positive',
    'other: weakly positive': 'positive'
}
geno_tmp.Genotoxicity[other_idx] = list(map(lambda x: other_map.get(x), geno_tmp.Genotoxicity[other_idx]))


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

# geno_consv = geno[['Chemical', 'CasRN']].drop_duplicates(['CasRN']).reset_index(drop = True)
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


# geno_maj = geno[['Chemical', 'CasRN']].drop_duplicates(['CasRN']).reset_index(drop = True)
result['maj'] = result.CasRN.map(lambda x: extract_majority_endpoint(x))
result.maj.isna().sum()
result.maj.notna().sum()

result.maj.value_counts()
result.maj.value_counts(normalize = True)


#%%
result.to_excel('../tg471.xlsx', index = False, header = True)
