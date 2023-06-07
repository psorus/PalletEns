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

    gx,gi,tx,ti=f["gx"],f["gi"],f["tx"],f["ti"]
    x,j=f["x"],f["i"]

    gx=np.array([transform(i) for i in tqdm(gx)])
    tx=np.array([transform(i) for i in tqdm(tx)])
    x=np.array([transform(i) for i in tqdm(x)])

    np.savez_compressed(outp,gx=gx,gi=gi,tx=tx,ti=ti,x=x,i=j)    





if __name__ == '__main__':
    fn="/main/admin/pallet_ensemble/tinydata.npz"
    outp="reproc.npz"
    process(fn,outp)
