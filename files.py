
image=["siamese502-kfold-on-0_28_all.npz","siamese502-kfold-on-1_30_all.npz","siamese502-kfold-on-2_16_all.npz","siamese502-kfold-on-3_8_all.npz","siamese502-kfold-on-4_2_all.npz"]
image=["image/"+zw for zw in image]
avgcol=["emb0.npz","emb1.npz","emb2.npz","emb3.npz","emb4.npz"]
multicol=avgcol
stdcol=avgcol
avgcol=["avg_col/"+zw for zw in avgcol]
multicol=["multi_color/"+zw for zw in multicol]
stdcol=["std_color/"+zw for zw in stdcol]


graph=[f"graph/siamese-gnn-fold-{i}_90_all.npz" for i in range(5)]


folders={"graph":graph,"stdcol":stdcol,"avgcol":avgcol,"multicol":multicol,"image":image}
basep="/main/admin/pallet_ensemble/"
folders={key:[basep+zw for zw in val] for key,val in folders.items()}

from isvalid import rename, valid

validfolders={rename(key):val for key,val in folders.items() if valid(key)}

if __name__=="__main__":
    print(validfolders.keys())
