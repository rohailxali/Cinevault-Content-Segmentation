import pandas as pd
import numpy as np

df = pd.read_csv('Dataset.csv')
print('=== SHAPE ===')
print(df.shape)
print()
print('=== COLUMNS ===')
print(df.columns.tolist())
print()
print('=== DTYPES ===')
print(df.dtypes)
print()
print('=== HEAD (3 rows) ===')
print(df.head(3).to_string())
print()
print('=== NULL COUNTS ===')
print(df.isnull().sum())
print()
print('=== BLANK STRINGS ===')
for col in df.select_dtypes(include='object').columns:
    blanks = (df[col].str.strip() == '').sum()
    if blanks > 0:
        print(f'{col}: {blanks} blank strings')
print()
print('=== PLACEHOLDER CHECK ===')
placeholder_kws = ['not given', 'unknown', 'n/a', 'none', 'na', 'null']
for col in df.select_dtypes(include='object').columns:
    for kw in placeholder_kws:
        count = df[col].str.lower().str.strip().eq(kw).sum()
        if count > 0:
            print(f'  {col}: {count} values matching placeholder "{kw}"')
print()
print('=== DUPLICATE ROWS ===')
print(f'Duplicate rows: {df.duplicated().sum()}')
print()
print('=== UNIQUE VALUE COUNTS (object cols) ===')
for col in df.select_dtypes(include='object').columns:
    print(f'  {col}: {df[col].nunique()} unique values')
print()
print('=== NUMERICAL STATS ===')
print(df.select_dtypes(include='number').describe().to_string())
print()
print('=== SAMPLE VALUES (object cols, top 5) ===')
for col in df.select_dtypes(include='object').columns:
    print(f'  {col}: {df[col].dropna().value_counts().head(5).to_dict()}')
print()
print('=== DURATION SAMPLE ===')
if 'duration' in df.columns:
    print(df['duration'].dropna().head(20).tolist())
    print('Duration value counts (top 20):')
    print(df['duration'].value_counts().head(20))
