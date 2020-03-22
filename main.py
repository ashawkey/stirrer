from common import *
from stirrer import *

from sklearn.metrics import f1_score 

def MacroF1(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(np.clip(preds, 0, 10)).astype(int)
    score = f1_score(labels, preds, average = 'macro')
    return ('MacroF1', score, True)

def add_features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)
    df.index = ((df.time * 10_000) - 1).values
    df['batch'] = df.index // 50_000
    df['batch_index'] = df.index  - (df.batch * 50_000)
    df['batch_slices'] = df['batch_index']  // 5_000
    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)
    
    for c in ['batch','batch_slices2']:
        d = {}
        d['mean'+c] = df.groupby([c])['signal'].mean()
        d['median'+c] = df.groupby([c])['signal'].median()
        d['max'+c] = df.groupby([c])['signal'].max()
        d['min'+c] = df.groupby([c])['signal'].min()
        d['std'+c] = df.groupby([c])['signal'].std()
        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))
        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))
        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))
        for v in d:
            df[v] = df[c].map(d[v].to_dict())
        df['range'+c] = df['max'+c] - df['min'+c]
        df['maxtomin'+c] = df['max'+c] / df['min'+c]
        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2
    
    #add shifts
    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])
    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]
    for i in df[df['batch_index']==0].index:
        df['signal_shift_+1'][i] = np.nan
    for i in df[df['batch_index']==49999].index:
        df['signal_shift_-1'][i] = np.nan

    # add shifts_2
    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])
    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0] + [1]
    for i in df[df['batch_index']==0].index:
        df['signal_shift_+2'][i] = np.nan
    for i in df[df['batch_index']==1].index:
        df['signal_shift_+2'][i] = np.nan
    for i in df[df['batch_index']==49999].index:
        df['signal_shift_-2'][i] = np.nan
    for i in df[df['batch_index']==49998].index:
        df['signal_shift_-2'][i] = np.nan 
        
    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:
        df[c+'_msignal'] = df[c] - df['signal']

    return df

if PREPROCESS:
    train = pd.read_csv('data/train_clean.csv')
    test = pd.read_csv('data/test_clean.csv')
    train = add_features(train).pipe(reduce_mem_usage)
    test = add_features(test).pipe(reduce_mem_usage)
    train.to_csv('data/train_preprocessed.csv', index=False)
    test.to_csv('data/test_preprocessed.csv', index=False)

features = [
    'meanbatch', 
    'medianbatch', 
    'maxbatch', 
    'minbatch', 
    'stdbatch', 
    'mean_abs_chgbatch', 
    'abs_maxbatch', 
    'abs_minbatch', 
    'rangebatch', 
    'maxtominbatch', 
    'abs_avgbatch', 
    'meanbatch_slices2', 
    'medianbatch_slices2', 
    'maxbatch_slices2', 
    'minbatch_slices2',
    'stdbatch_slices2',
    'mean_abs_chgbatch_slices2',
    'abs_maxbatch_slices2',
    'abs_minbatch_slices2',
    'rangebatch_slices2',
    'maxtominbatch_slices2',
    'abs_avgbatch_slices2',
    'signal_shift_+1',
    'signal_shift_-1',
    'signal_shift_+2',
    'signal_shift_-2',
    'meanbatch_msignal',
    'medianbatch_msignal',
    'maxbatch_msignal',
    'minbatch_msignal',
    'stdbatch_msignal',
    'mean_abs_chgbatch_msignal',
    'abs_maxbatch_msignal',
    'abs_minbatch_msignal',
    'rangebatch_msignal',
    'maxtominbatch_msignal',
    'abs_avgbatch_msignal',
    'meanbatch_slices2_msignal',
    'medianbatch_slices2_msignal',
    'maxbatch_slices2_msignal',
    'minbatch_slices2_msignal',
    'stdbatch_slices2_msignal',
    'mean_abs_chgbatch_slices2_msignal',
    'abs_maxbatch_slices2_msignal',
    'abs_minbatch_slices2_msignal',
    'rangebatch_slices2_msignal',
    'maxtominbatch_slices2_msignal',
    'abs_avgbatch_slices2_msignal',
    'signal_shift_+1_msignal',
    'signal_shift_-1_msignal',
    'signal_shift_+2_msignal',
    'signal_shift_-2_msignal',
]


if TRAIN:
    if not PREPROCESS:
        train = pd.read_csv('data/train_preprocessed.csv')
    X = train[features]
    y = train['open_channels'].astype(int)
    print(X.shape, y.shape)

    trainer = Trainer(
            workspace, 
            num_boost_round=1000, 
            feval=MacroF1,
            backend='lgb',	
        )

    idx = np.arange(X.shape[0])
    kf = model_selection.KFold(num_splits, shuffle=True, random_state=seed)
    for fold in range(num_splits):
        train_idx, val_idx = list(kf.split(idx))[fold]
        trainer.model_name = str(fold) # for convenience
        trainer.train(X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx], y.iloc[val_idx])

if PREDICT:
    if not PREPROCESS:
        test = pd.read_csv('data/test_preprocessed.csv')
    predictor = Predictor(workspace)

    for fold in range(num_splits):
        predictor.load_checkpoint(f"{workspace}/{fold}.txt")

    X = test[features]
    pred = predictor.predict(X)
    pred = np.round(np.clip(pred, 0, 10)).astype(int)
    sub = pd.read_csv("data/sample_submission.csv", dtype={'time':str})
    sub.iloc[:,1] = pred
    sub.to_csv("pred.csv", index=False)
    
    """
    plt.figure(figsize=(20,5))
    res = 1000
    let = ['A','B','C','D','E','F','G','H','I','J']
    plt.plot(range(0,test.shape[0],res),sub.open_channels[0::res])
    for i in range(5): plt.plot([i*500000,i*500000],[-5,12.5],'r')
    for i in range(21): plt.plot([i*100000,i*100000],[-5,12.5],'r:')
    for k in range(4): plt.text(k*500000+250000,10,str(k+1),size=20)
    for k in range(10): plt.text(k*100000+40000,7.5,let[k],size=16)
    plt.title('Test Data Predictions',size=16)
    plt.show()
    """	    
