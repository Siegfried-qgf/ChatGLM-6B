from datasets import load_dataset
data_files = {}
cache_dir='/workspace/cache'
data_path='/workspace/ChatGLM-6B/dataset/ultrachat/'
data_files["test"] = data_path+'train_0.json'
raw_datasets = load_dataset(
    'json',
    data_files=data_files,
    cache_dir=cache_dir,
)
l=[]
for i in range(101):
    l.append(raw_datasets['test'][i])
import json
with open('./examples.json','w') as f:
    json.dump(l, f)