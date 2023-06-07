import numpy as np
from tqdm import tqdm

clusters=16
overlap=8


def iterate_image(img):
    for i in range(0,img.shape[0],clusters-overlap):
        for j in range(0,img.shape[1],clusters-overlap):
            yield img[i:i+clusters,j:j+clusters,:]


def transform(arr):
    #gets image of size 128, 384, 3

    return np.array([np.std(zw,axis=(0,1)) for zw in iterate_image(arr)]).flatten()

def process(fil,outp):
    f=np.load(fil)

    x,fn,i,gx,gn,gi,tx,tn,ti,new,newf,newi=f['x'],f['fn'],f['i'],f['gx'],f['gn'],f['gi'],f['tx'],f['tn'],f['ti'],f['new'],f['newf'],f['newi']

    x=np.array([transform(i) for i in tqdm(x)])
    gx=np.array([transform(i) for i in tqdm(gx)])
    tx=np.array([transform(i) for i in tqdm(tx)])
    new=np.array([transform(i) for i in tqdm(new)])

    np.savez_compressed(outp,x=x,fn=fn,i=i,gx=gx,gn=gn,gi=gi,tx=tx,tn=tn,ti=ti,new=new,newf=newf,newi=newi)





if __name__ == '__main__':
    for i in range(5):
        fn = f"/global/splits/light_on_{i}.npz"
        outp=f"proc{i}.npz"
        process(fn,outp)
