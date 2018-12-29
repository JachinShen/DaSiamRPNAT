import scipy.io as sio
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description="your script description")
parser.add_argument('--input', '-i', required=True, help='input json file')
parser.add_argument('--name', '-n', required=True, help='model name')
args = parser.parse_args()
sequences = open("../SEQUENCES", "r")
seq = sequences.readline()
# res = json.load(open("./DaSiamRPNOTB.json", "r"))
res = json.load(open(args.input, "r"))
result_path = "../otb-toolkit/results/OPE/"
while seq:
    seq = seq[:-1]
    print(seq)
    matfn = result_path + "{}_MDNet.mat".format(seq)
    data = sio.loadmat(matfn)
    data["results"][0][0][0][0][0] = res[seq]
    sio.savemat(result_path+"{}_{}.mat".format(seq, args.name), data)

    seq = sequences.readline()

print("Convert json from: {} to mat of model: {} successfully".format(args.input, args.name))
# print(data["results"][0][0][0][0][1])
