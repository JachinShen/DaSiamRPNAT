import scipy.io as sio
import numpy as np
import json
sequences=open("../../sequences/SEQUENCES", "r")
seq=sequences.readline()
res=json.load(open("../../trackers/Example/DaSiamRPNAT.json", "r"))
while seq:
    seq=seq[:-1]
    print(seq)
    matfn="./{}_MDNet.mat".format(seq)
    data=sio.loadmat(matfn)
    data["results"][0][0][0][0][0]=res[seq]
    sio.savemat("{}_DaSiamRPNAT.mat".format(seq), data)

    seq=sequences.readline()

#print(data["results"][0][0][0][0][1])
print(data["results"])
