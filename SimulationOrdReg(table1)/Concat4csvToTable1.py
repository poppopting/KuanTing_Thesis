import pandas as pd

table1 = pd.concat([pd.read_csv('ToConcatenate\SGD_df.csv'),
                    pd.read_csv('ToConcatenate\ASGDandSGD_with_GammaDecay_df.csv'),
                    pd.read_csv('ToConcatenate\Batch_SGD_df.csv'),
                    pd.read_csv('ToConcatenate\Batch_ASGDandSGD_with_GammaDecay_df.csv'),
                    ], axis=0).rename(columns={'Unnamed: 0': 'Method'})

table1.to_csv('Table1.csv', index=False)
