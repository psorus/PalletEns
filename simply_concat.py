import numpy as np

from files import *
from files import validfolders as folders
import json

#del folders["imageAnh"]
#del folders["graph"]

#folders={"graph":folders["graph"],"newAnh":folders["newAnh"]}

#for each test image, find the closest galery image of the same type

def calc_accuracy(temb,tid,gemb,gid):
    temb,tid,gemb,gid=gemb,gid,temb,tid
    
    firstmatch=[]
    
    for tt,ti in zip(temb,tid):
        dist=(tt-gemb)**2
        while len(dist.shape)>1:
            dist=np.sum(dist,axis=-1)
    
        indice=np.argsort(dist)
        for i,ind in enumerate(indice):
            if gid[ind]==ti:
                firstmatch.append(i)
                break
    
    firstmatch=np.array(firstmatch)
    
    ranks=[1,2,3,4,5,6,7,8,9,10]
    
    ret={}
    for rank in ranks:
        ret[rank]=np.mean(firstmatch<rank)
    return ret


def file_to_acc(fn):
    f=np.load(fn)
    
    temb,tid,tfn,gemb,gid,gfn=f['temb'],f['tid'],f['tfn'],f['gemb'],f['gid'],f['gfn']

    return calc_accuracy(temb,tid,gemb,gid)

def prep(q):
    if len(q.shape)>2:
        q=np.mean(q,axis=1)
    return q
def concat_to_acc(fns):
    fs=[np.load(fn) for fn in fns]
    #print([f.files for f in fs])
    #print([f['temb'].shape for f in fs])
    emb=np.concatenate([prep(f['emb']) for f in fs],axis=-1)
    mn,std=np.mean(emb,axis=0),np.std(emb,axis=0)+1e-6
    temb=np.concatenate([prep(f['temb']) for f in fs],axis=-1)
    gemb=np.concatenate([prep(f['gemb']) for f in fs],axis=-1)
    temb=(temb-mn)/std
    gemb=(gemb-mn)/std
    tid,tfn,gid,gfn=fs[0]['tid'],fs[0]['tfn'],fs[0]['gid'],fs[0]['gfn']
    return calc_accuracy(temb,tid,gemb,gid)

def average_dics(dics):
    keys=dics[0].keys()
    ret={}
    for key in keys:
        ret[key]=np.mean([dic[key] for dic in dics])
    return ret

def std_dics(dics):
    keys=dics[0].keys()
    ret={}
    for key in keys:
        ret[key]=np.std([dic[key] for dic in dics])
    return ret

def stats_dics(dics):
    keys=dics[0].keys()
    ret={}
    for key in keys:
        a,b=np.mean([dic[key] for dic in dics]),np.std([dic[key] for dic in dics])
        ret[key]=f"{a:.3f}+-{b:.3f}"
    return ret

lis=[]
for i in range(5):#5 folds
    fns=[folders[key][i] for key in folders]
    lis.append(concat_to_acc(fns))

print(stats_dics(lis))


import sys
nam=sys.argv[0]
nam=nam[nam.rfind('/')+1:]
nam=nam.replace('.py','')
with open(f"results/{nam}.json",'w') as f:
    json.dump(stats_dics(lis),f,indent=2)

arr=[[li[i] for i in range(1,11)] for li in lis]
arr=np.array(arr)

np.savez_compressed(f"partials/{nam}.npz",q=arr)


