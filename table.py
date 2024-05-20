import pandas as pd

columns = ['HitRate@50', 'MRR@50', 'Precision@50', 'Recall@50', 'NDCG@50']


shallow_map = {
    'musicnn': 'musicnn-Apr16_17:36_0_20_nc',
    'musicfm': 'musicfm-Apr19_00:07_0_20_nc',
    'music2vec': 'music2vec-Apr16_17:42_0_20_nc',
    'encodecmae': 'encodecmae-Apr19_00:07_0_20_nc',
    'mert': 'mert-Apr19_00:07_0_20_nc',
    'jukemir': 'jukemir-Apr19_00:11_0_20_nc',
    'mfcc': 'mfcc-May09_21:19_0_20_nc',
    # 'mfcc_bow': 'mfcc_bow-May09_21:19_0_20_nc'
}

cosine_map = {
    'musicnn': 'musicnn_cosine',
    'musicfm': 'musicfm_cosine',
    'music2vec': 'music2vec_cosine',
    'encodecmae': 'encodecmae_cosine',
    'mert': 'mert_cosine',
    'jukemir': 'jukemir_cosine',
    'mfcc': 'mfcc_cosine',
    # 'mfcc_bow': 'mfcc_bow_cosine'
}

bert_map = {
    'musicnn': 'bert_musicnn-Apr29_21:57_0_300_nc',
    'musicfm': 'bert_musicfm-Apr29_19:52_0_300_nc',
    'music2vec': 'bert_music2vec-Apr29_19:57_0_300_nc',
    'encodecmae': 'bert_encodecmae-Apr29_20:54_0_300_nc',
    'mert': 'bert_mert-Apr29_19:52_0_300_nc',
    'jukemir': 'bert_jukemir-Apr29_19:24_0_300_nc',
    'mfcc': 'bert_mfcc-May10_18:09_0_300_80g',
    # 'mfcc_bow': 'bert_mfcc_bow-May10_18:09_0_300_80g'
}

size_map = {
    'musicnn': 200,
    'musicfm': 750,
    'music2vec': 768,
    'encodecmae': 768,
    'mert': 1024,
    'jukemir': 4800,
    # 'mfcc_bow': 500,
    'mfcc': 104
}

# random_bert: 'bert_random-Apr26_20:10_128_300_nc'
# random_shallow: 'random-May02_23:47_128_20_shallow4ppr'

def read_res(model_map, round=3):
    res = pd.DataFrame(columns=columns + ['Size'])
    for model, run in model_map.items():
        df = pd.read_csv(f'metrics/{run}_test.csv', index_col=0).loc['mean'].round(round)
        df['Size'] = size_map[model]
        res.loc[model] = df
    res['Size'] = res['Size'].astype(int)
    return res

cs = read_res(cosine_map)
sr = read_res(shallow_map)
br = read_res(bert_map)

cs = cs.sort_values('HitRate@50')
cs.to_csv('cosine.csv')

sr = sr.loc[cs.index]
sr.to_csv('shallow.csv')

br = br.loc[cs.index]
br.to_csv('bert.csv')

print('cosine:\n', cs, '\n')
print('shallow:\n', sr, '\n')
print('bert4rec:\n', br, '\n')