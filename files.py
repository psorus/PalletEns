
imgAnh=["siamese502-kfold-on-0_28_all.npz","siamese502-kfold-on-1_30_all.npz","siamese502-kfold-on-2_16_all.npz","siamese502-kfold-on-3_8_all.npz","siamese502-kfold-on-4_2_all.npz"]
imgAnh=["imageAnh/"+zw for zw in imgAnh]
graph=["siamese-gnn-fold-0_67_all.npz","siamese-gnn-fold-1_67_all.npz","siamese-gnn-fold-2_67_all.npz","siamese-gnn-fold-3_67_all.npz","siamese-gnn-fold-4_67_all.npz"]
graph=["graph/"+zw for zw in graph]
avgcol=["emb0.npz","emb1.npz","emb2.npz","emb3.npz","emb4.npz"]
multicol=avgcol
stdcol=avgcol
avgcol=["avg_col/"+zw for zw in avgcol]
multicol=["multi_color/"+zw for zw in multicol]
stdcol=["std_color/"+zw for zw in stdcol]


medium=[f"gnn_medium/siamese-gnn-fold-{i}_90_all.npz" for i in range(5)]
smaller=[f"gnn_smaller/siamese-gnn-fold-{i}_54_all.npz" for i in range(5)]
newAnh=[f"newAnh/siamese502-kfold-on-{i}_30_all.npz" for i in range(5)]

latent=[f"latent/data_{i}.npz" for i in range(5)]

folders={"imageAnh":imgAnh,"graph":graph,"stdcol":stdcol,"avgcol":avgcol,"multicol":multicol,"medium":medium,"smaller":smaller,"newAnh":newAnh,"latent":latent}
basep="/main/admin/pallet_ensemble/"
folders={key:[basep+zw for zw in val] for key,val in folders.items()}

from isvalid import rename, valid

validfolders={rename(key):val for key,val in folders.items() if valid(key)}

if __name__=="__main__":
    print(validfolders.keys())
