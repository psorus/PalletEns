import numpy as np
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

task=0
if len(sys.argv)>1:
    task=int(sys.argv[1])

f=np.load(f"proc{task}.npz")

x,fn,i,gx,gn,gi,tx,tn,ti,new,newf,newi=f['x'],f['fn'],f['i'],f['gx'],f['gn'],f['gi'],f['tx'],f['tn'],f['ti'],f['new'],f['newf'],f['newi']

triplets=10000
repre=50
alpha=1.0

def build_triplets(x,i):
    #generate triplets many groups of three elements of x. Where the first one is random. the second one has the same value of i, while the last one has not

    #create a dictionary matching the values of i to the indices of x
    d={}
    for j in range(len(i)):
        if i[j] in d:
            d[i[j]].append(j)
        else:
            d[i[j]]=[j]

    poskey=list(d.keys())

    #create the triplets
    t=[]
    for j in range(triplets):
        block12,block3=np.random.choice(poskey,2,replace=False)

        dex1,dex2=np.random.choice(d[block12],2,replace=False)
        dex3=np.random.choice(d[block3])
        t.append(np.array([x[dex1],x[dex2],x[dex3]]))
    t=np.array(t)
    return t

t=build_triplets(x,i)
print(t.shape)

ishape=t.shape[2:]
inp=keras.layers.Input(ishape)
q=inp
q=keras.layers.Dense(100,activation='relu')(q)
q=keras.layers.Dense(100,activation='relu')(q)
q=keras.layers.Dense(repre,activation='relu')(q)

model=keras.Model(inp,q)

inp=keras.layers.Input(t.shape[1:])
q=inp
q1=model(q[:,0])
q2=model(q[:,1])
q3=model(q[:,2])

#expand dims
q1=K.expand_dims(q1,1)
q2=K.expand_dims(q2,1)
q3=K.expand_dims(q3,1)

q=keras.layers.concatenate([q1,q2,q3],1)

dT=K.mean(K.square(q1-q2),axis=(1,2))
dF=K.mean(K.square(q1-q3),axis=(1,2))
loss=K.mean(K.maximum(0.0,dT-dF+alpha))


hmodel=keras.Model(inp,q)
hmodel.add_loss(loss)
hmodel.compile(optimizer='adam')
hmodel.summary()


hmodel.fit(t,epochs=1000,verbose=1,validation_split=0.1,batch_size=100,shuffle=True,callbacks=[keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)])


emb=model.predict(x)
temb=model.predict(tx)
gemb=model.predict(gx)
newemb=model.predict(new)


np.savez_compressed(f"emb{task}.npz",emb=emb,temb=temb,gemb=gemb,newemb=newemb,fn=fn,tfn=tn,gfn=gn,newfn=newf,id=i,tid=ti,gid=gi,newid=newi)






