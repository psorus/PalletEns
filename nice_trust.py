import numpy as np
from plt import *

from files import validfolders as folders
from files import folders
from files import rename
import sys

from PIL import Image

plt.figure(figsize=(14,16))

baseid=28
if len(sys.argv)>1:
    baseid=int(sys.argv[1])
fold=1
if len(sys.argv)>2:
    fold=int(sys.argv[2])

f=np.load(f"/global/splits/light_on_{fold}.npz")
gx,tx,gi,ti=f["gx"],f["tx"],f["gi"],f["ti"]
fn,gn,tn=f["fn"],f["gn"],f["tn"]

f.close()

methods=["newAnh","medium","stdcol"]

def img_from_disk(fn):
    fn=fn[fn.find("/")+1:]
    fn="require/"+fn

    #print("trying to read from disk",fn)

    im=Image.open(fn)
    im=im.resize((1800,1000))
    ac=np.asarray(im)
    return ac



def load_image(fn,img):
    try:
        return img_from_disk(fn+".jpg")
    except:
        pass
    try:
        return img_from_disk(fn+".bmp")
    except:
        pass
    print(fn)
    return img


cou=4


sub=plt.subplot(4+2*cou,3,(1,9))
def drawone(img,sub,rid,shallcolor=True):
    plt.imshow(img)#,aspect="auto")
    plt.axis("off")
    ax=sub.axis()
    if rid==realid:
        col,alpha="darkgreen",0.3
    else:
        col,alpha="darkred",0.2
    if shallcolor:
        alpha=1.0
        rec=plt.Rectangle((ax[0],ax[2]),ax[1]-ax[0],ax[3]-ax[2],color=col,alpha=alpha,fill=None,lw=10)
        sub.add_patch(rec)
    #fill box
    plt.annotate(f"{rid}",(150 if shallcolor else 50,820 if shallcolor else 930),color="white",fontsize=20,bbox=dict(facecolor=col if shallcolor else "black",alpha=0.7))

    plt.gca().patch.set_edgecolor('purple')
    plt.gca().patch.set_linewidth('5')


    #plt.annotate(f"{rid}",(ax[0],ax[2]),color="white",fontsize=20)

realid=gi[baseid]

drawone(load_image(gn[baseid],gx[baseid]),sub,realid,shallcolor=False) 

orders=[[0,1,3,4],
        [0,1,2,3],
        [1,3,12,20]]

#relatives=[[tx[0],tx[1],tx[2],tx[12]],
#           [tx[3],tx[4],tx[5],tx[13]],
#           [tx[6],tx[7],tx[8],tx[14]]]



#relaid=[[ti[0],ti[1],ti[2],ti[12]],
#        [ti[3],ti[4],ti[5],ti[13]],
#        [ti[6],ti[7],ti[8],ti[14]]]

ax=plt.gca()
ax.annotate('', xy=(-0.52, -0.1), xycoords='axes fraction', xytext=(1.52, -0.1),arrowprops=dict(arrowstyle="-", color='black',lw=3))

col="purple"
#put text "query" next to the first img, bold text
#ax.annotate('Query', xy=(-0., +0.5), xycoords='axes fraction', xytext=(-0.38, +0.1),fontsize=30,color=col,arrowprops=dict(arrowstyle="->", color=col,lw=3))#,fontweight="bold")

#ax.annotate('a)', xy=(-0., +0.5), xycoords='axes fraction', xytext=(-0.18, +0.47),fontsize=35,color="black")#col,bbox=dict(facecolor='none', edgecolor=col))
#ax.annotate('A', xy=(-0., +0.5), xycoords='axes fraction', xytext=(-0.18, +0.4),fontsize=30,color=col,bbox=dict(facecolor='none', edgecolor=col))

orders=[]
for methodi,method in enumerate(methods):
    fn=folders[method][fold]
    f=np.load(fn)
    cgx,ctx=f["gemb"],f["temb"]
    truerep=cgx[baseid]
    dist=np.square(ctx-truerep)
    while len(dist.shape)>1:
        dist=np.mean(dist,axis=1)
    #dist=np.mean(np.square(ctx-truerep),axis=(1))
    order=np.argsort(dist)
    print(order[:cou])
    orders.append(order[:cou])

relatives=[[load_image(tn[iii],tx[iii]) for iii in ii] for ii in orders]
relaid=[[ti[iii] for iii in ii] for ii in orders]

dex=7#+3
for indi in range(len(relatives[0])):
    for algoi in range(len(relatives)):
        #print(dex)
        sub=plt.subplot(cou+2,3,dex)
        if not indi:
            plt.title(rename(methods[algoi]).replace(" based",""))
       # plt.title(f"{algoi} {indi}")
        drawone(relatives[algoi][indi],sub,relaid[algoi][indi])
        #plt.imshow(relatives[algoi][indi])
        #plt.bar(np.arange(0,10),np.random.randint(0,10,10),color="goldenrod")
        #plt.axis("off")
        plt.gca().zorder=0
        dex+=1
    #dex+=2
#plt.tight_layout()

print(np.sum(np.array(relaid)==realid))


plt.subplot(4+2*cou,3,(1,9))

#ax.annotate('b)', xy=(-0., -0.5), xycoords='axes fraction', xytext=(-0.600, -1.835),zorder=+1000,fontsize=35,color="black",bbox=dict(facecolor='grey', edgecolor="none",zorder=999))
#ax.annotate('b)', xy=(-0., -0.5), xycoords='axes fraction', xytext=(0.100, -0.335),fontsize=35,color="black")#,bbox=dict(facecolor='none', edgecolor=col))
#ax.annotate('B', xy=(-0., -0.5), xycoords='axes fraction', xytext=(0.11, -0.32),fontsize=30,color=col,bbox=dict(facecolor='none', edgecolor=col))
#ax.annotate('B', xy=(-0., -0.5), xycoords='axes fraction', xytext=(-0.0, -0.35),fontsize=30,color=col,arrowprops=dict(arrowstyle="->", color=col,lw=3),bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=3'))


plt.savefig("imgs/nice_trust.png",format="png")
plt.savefig("imgs/nice_trust.pdf",format="pdf")










