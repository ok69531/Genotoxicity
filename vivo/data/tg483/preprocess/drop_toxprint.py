import pandas as pd

smiles_df = pd.read_excel('../tg483_wSMILES_ToxPrint_desalt.xlsx')
smiles_df = smiles_df.drop(['DTXSID'], axis = 1)

cols = list(smiles_df.columns)
cols[:3] = ['CasRN', 'Chemical', 'SMILES']
smiles_df.columns = cols


cols_front = ['Chemical', 'CasRN', 'SMILES', 'consv', 'maj']
smiles_df = smiles_df[[c for c in cols_front if c in smiles_df] + [c for c in smiles_df if c not in cols_front]]

df = smiles_df[smiles_df.iloc[:, 6].notna()].reset_index(drop = True)

print(df.consv.value_counts())
print(df.consv.value_counts(normalize=True))

print(df.maj.value_counts())
print(df.maj.value_counts(normalize=True))

df.to_excel('../tg483.xlsx', header = True, index = False) 

