import scipy.io as sio
import numpy as np
import json
sequences=open("../SEQUENCES", "r")
seq=sequences.readline()
res=json.load(open("./DaSiamRPNAT.json", "r"))
result_path = "../otb-toolkit/results/OPE/"
while seq:
    seq=seq[:-1]
    print(seq)
    matfn=result_path + "{}_MDNet.mat".format(seq)
    data=sio.loadmat(matfn)
    data["results"][0][0][0][0][0]=res[seq]
    sio.savemat(result_path+"{}_DaSiamRPNAT.mat".format(seq), data)

    seq=sequences.readline()

print("Convert json to mat successfully")
#print(data["results"][0][0][0][0][1])
#print(data["results"])
