import numpy as np

from files import *
from files import validfolders as folders

from scipy.optimize import minimize

from tqdm import tqdm
import json

from flaml import tune


#del folders["imageAnh"]
#del folders["graph"]


triplets=1000
alpha=1.0


def quick_rank1(temb,tid,gemb,gid):
    temb,tid,gemb,gid=gemb,gid,temb,tid
    #print(temb.shape,gemb.shape)
    #exit()
    val=0
    for tt,ti in zip(temb,tid):
        dist=(tt-gemb)**2
        while len(dist.shape)>1:
            dist=np.sum(dist,axis=-1)
        mind=np.argmin(dist)
        if gid[mind]==ti:
            val+=1
    return val/len(tid)


def quick_evo(func,init,n=1000):
    curr=init
    best=func(curr)
    for i in tqdm(range(n),total=n):
        new=np.copy(curr)
        new[np.random.randint(len(new))]+=np.random.normal(0,np.exp(-np.random.uniform(0,10)))
        #new=curr+np.random.normal(size=len(curr))
        new_val=func(new)
        if new_val>best:
            curr=new
            best=new_val
    return curr



def modify(data):
    if len(data.shape)>2:
        data=data.reshape(data.shape[0],-1)
    return data


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

def test_galery_split(x,i):
    iss=np.unique(i)
    np.random.shuffle(iss)
    dic={ii:[] for ii in iss}
    for xx,ii in zip(x,i):
        dic[ii].append(xx)
    for ii in iss:
        np.random.shuffle(dic[ii])
    tx,ti,gx,gi=[],[],[],[]
    for ii in iss:
        gx.append(dic[ii][0])
        gi.append(ii)
        for zw in dic[ii][1:]:
            tx.append(zw)
            ti.append(ii)

    tx=np.array(tx)
    gx=np.array(gx)
    ti=np.array(ti)
    gi=np.array(gi)
    return tx,ti,gx,gi


def create_flaml_vector(n):
    return {f"v{i}":tune.loguniform(0.01,100) for i in range(n)}
def flaml_dic_to_vector(dic):
    keys=list(dic.keys())
    return np.array([dic[key] for key in [f"v{i}" for i in range(len(keys))] if key in keys])



def do_fold(iteration):

    i,gi,ti=None,None,None
    fn,gfn,tfn=None,None,None
    x,gx,tx=[],[],[]
    repeats=[]
    for folder, files in folders.items():
        file=files[iteration]
        f=np.load(file)
        print(file)
        if i is None:
            i=f['id']
        if gi is None:
            gi=f['gid']
        if ti is None:
            ti=f['tid']
        if fn is None:
            fn=f['fn']
        if gfn is None:
            gfn=f['gfn']
        if tfn is None:
            tfn=f['tfn']
        x.append(modify(f['emb']))
        gx.append(modify(f['gemb']))
        tx.append(modify(f['temb']))
        repeats.append(x[-1].shape[1])
    x=np.concatenate(x,axis=-1)
    gx=np.concatenate(gx,axis=-1)
    tx=np.concatenate(tx,axis=-1)
    mn,st=np.mean(x,axis=0),np.std(x,axis=0)
    st+=1e-10
    #mn,st=0.0,1.0
    x=(x-mn)/st
    gx=(gx-mn)/st
    tx=(tx-mn)/st

    xt,xti,xg,xgi=test_galery_split(x,i)

    init=np.ones(len(repeats))/len(repeats)

    def loss(q):
        base,true,false=q[:,0],q[:,1],q[:,2]
        dT=np.mean((base-true)**2,axis=-1)
        dF=np.mean((base-false)**2,axis=-1)
        return np.maximum(0,dT-dF+alpha)

    def adaptw(w):
        w=np.abs(w)
        w=w/np.sum(w)
        return w
    
    def transform_weights(w):
        ret=[]
        for ww,r in zip(w,repeats):
            for i in range(r):
                ret.append(ww)
        return np.array(ret)

    def eval_weights(w):
        w=adaptw(w)
        w=transform_weights(w)
        #w=np.array(w)
        #data=np.dot(train,w)
        data=train*w
        return np.mean(loss(data))
    def weighted_rank1(temb,tid,gemb,gid):
        def func(weig):
            if type(weig) is dict:
                weig=flaml_dic_to_vector(weig)
            w=adaptw(weig)
            weig=transform_weights(w)
            return {"score":quick_rank1(temb*weig,tid,gemb*weig,gid)}
        return func

    def optimal_weights(temb,tid,gemb,gid):
        #init=np.ones(temb.shape[-1])
        hyper=create_flaml_vector(len(repeats))
        #init=np.ones(len(repeats))
        func=weighted_rank1(temb,tid,gemb,gid)
        #res=minimize(func,init,method='Nelder-Mead',options={'disp': True}).x
        res=tune.run(func,config=hyper,metric="score",mode="max",num_samples=3000000,time_budget_s=60,resources_per_trial={"cpu":10}).best_config
        #res=tune.run(func,config=hyper,metric="score",mode="max",num_samples=3000000,time_budget_s=250*60,resources_per_trial={"cpu":10}).best_config
        #res=tune.run(func,config=hyper,metric="score",mode="max",num_samples=3,time_budget_s=150*60).best_config
        res=flaml_dic_to_vector(res)
        #res=quick_evo(func,init)
        return transform_weights(np.array(res))


    #wei=minimize(eval_weights,init,method='Nelder-Mead',options={'disp': True}).x
    wei=optimal_weights(xt,xti,xg,xgi)
    wei=adaptw(wei)
    print(wei)
    #wei=transform_weights(wei)
    #print(wei)

    gx=gx*wei
    tx=tx*wei
    
    #print(eval_weights(init))

    acc=calc_accuracy(tx,ti,gx,gi)
    print(acc)

    with open(f"results_newflaml_{iteration}","w") as f:
        f.write(json.dumps({"acc":acc,"wei":wei.tolist(),"iter":iteration},indent=2))
    return acc





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
#for i in range(1):#debug
for i in range(5):#5 folds
    lis.append(do_fold(i))

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

