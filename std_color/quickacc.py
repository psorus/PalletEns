import numpy as np


def calc_accuracy(temb,tid,gemb,gid):
    
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

if __name__=='__main__':
    fn="emb0.npz"
    fn="retrainemb0.npz"
    f=np.load(fn)
    temb,tid,gemb,gid=f['temb'],f['tid'],f['gemb'],f['gid']
    print(calc_accuracy(temb,tid,gemb,gid))


