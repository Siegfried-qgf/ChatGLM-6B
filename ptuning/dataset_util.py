from datasets import load_dataset
import json 
l=[]
def ultrachat(banben):
    global l
    source_p='/workspace/ChatGLM-6B/dataset/ultrachat/train_{:d}.jsonl'.format(banben)
    with open(source_p,'r') as f:
        for line in f:
            try:
                dic = json.loads(line)
                for i in dic['data']:
                    l.append({'id':dic['id'],'data':i})
            except:
                continue
        print(str(banben)+" :"+str(len(l)))
'''
0 :1637156
1 :3335064
2 :4971900
3 :6405332
4 :7291618
5 :8129550
6 :8966238
7 :10099444
8 :11299222
'''
if __name__=='__main__':
    target_p='/workspace/ChatGLM-6B/dataset/ultrachat/train.json'
    for i in range(0,9):
        ultrachat(i)
    with open(target_p,'w') as f:
        json.dump(l, f)
    
