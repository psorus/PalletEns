import numpy as np
from files import *
from files import folders

takeonly=None
#takeonly="stdcol"
if not takeonly is None:
    folders={takeonly:folders[takeonly]}

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

dic={}
for fold,names in folders.items():
    print(fold)
    lis=[]
    for name in names:
        try:
            acc=file_to_acc(name)
            lis.append(acc)
        except:
            print("error",name)
    print(stats_dics(lis))
    dic[fold]=stats_dics(lis)

import json
import sys
nam=sys.argv[0]
nam=nam[nam.rfind('/')+1:]
nam=nam.replace('.py','')
with open(f"results/{nam}.json",'w') as f:
    json.dump(dic,f,indent=2)




