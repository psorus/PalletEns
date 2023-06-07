
remeth={"newAnh":"Image based",
        "medium":"Graph based",
        "multicol":"Average color based",
        "stdcol":"Color variance based",
        "avgcol":"Brightness based"
        }

reens={"newflamlize":"Weighted Rank-1",
       "simply_concat":"Concatenation",
       "nn_concat":"NN Triplet",
       "majority_vote":"Majority Vote",
       "sub_weighted":"Weighted Triplet"
       }

def valid(x):
    return x in remeth.keys() or x in reens.keys()

def rename(x):
    if x in remeth.keys():
        return remeth[x]
    elif x in reens.keys():
        return reens[x]
    else:
        return x

def unname(x):
    for key,val in remeth.items():
        if val==x:
            return key
    for key,val in reens.items():
        if val==x:
            return key
    return x

def whatami(x):
    if x in remeth.keys():
        return "method"
    elif x in reens.keys():
        return "ensemble"
    else:
        for key,val in remeth.items():
            if val==x:
                return "method"
        for key,val in reens.items():
            if val==x:
                return "ensemble"
        return "unknown"


namtocol={"newAnh":"#845B97",
          "medium":"#00B945",
          "multicol":"#FF9500",
          "stdcol":"#0C5DA5",
          "avgcol":"#FF2C00"
          }

enstocol={"newflamlize":"#845B97",
          "simply_concat":"#00B945",
          "nn_concat":"#FF9500",
          "majority_vote":"#0C5DA5",
          "sub_weighted":"#FF2C00"
          }

def nametocolor(x):
    if x in remeth.keys():
        return namtocol[x]
    elif x in reens.keys():
        return enstocol[c]
    else:
        for key,val in remeth.items():
            if val==x:
                return namtocol[key]
        for key,val in reens.items():
            if val==x:
                return enstocol[key]
        return "#000000"




if __name__=="__main__":
    from plt import plt

    plt.bar(range(len(rename)),[1]*len(rename),color=[nametocolor(x) for x in rename.keys()])
    plt.savefig("debug.png")
    plt.show()


