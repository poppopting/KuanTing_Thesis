import pandas as pd

def rec_train_test_split(data, size=0.2):
    data_len = len(data)
    rows_to_sample = round(data_len * size)
    test_df = pd.DataFrame(columns = ['user', 'item', 'ratings'])
    train_df = data.copy()
    while len(test_df) != rows_to_sample:
        print('='*15 + 'Begin sampling'+ '='*15)
        rows_still_sample = rows_to_sample - len(test_df)
        
        new_test = train_df.sample(n=rows_still_sample)
        test_df = pd.concat([test_df, new_test], axis=0, sort=False)
        
        train_df = train_df.drop(new_test.index)
        test_df = test_df[(test_df['user'].isin(train_df['user'])) &
                          (test_df['item'].isin(train_df['item']))]
        train_df = data.drop(test_df.index)
        print('We have sampled {0:,} / {1:,}'.format(len(test_df), rows_to_sample))
        print('Still need to sample {0:,}'.format(rows_to_sample - len(test_df)))
    print('Split completed !!!')
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
