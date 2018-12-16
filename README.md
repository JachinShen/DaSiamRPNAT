# DaSiamRPNAT

## Prerequisities

- python3
- pytorch
- numpy
- opencv
- matlab

## Data Preparation

### download OTB dataset

```
cd dataset
./download.sh
```

### extract zips

```
cd dataset
./extract.sh
```

### collect image path and ground truth

```
python pre_data_otb.py
```

Then, a file named `vot-otb.pkl` will appear in the folder `data`. It contains the image file path and ground truth.

## Pretrained model

SiamRPNBIG.model: https://drive.google.com/file/d/1-vNVZxfbIplXHrqMHiJJYWXYWsOIvGsf/view?usp=sharing

## run

```
cd code
python main.py
```

Then, a file named `DaSiamRPNAT.json` will appear. It records the results.

And some pictures will appear in folder `figure` to show the results.

## benchmark

### convert json to mat

```
cd code
python generate_mat.py
```

### run benchmark

In the folder `otb-toolkit`, run matlab. Then, run `run_OPE` in matlab.

## Reference

- DaSiamRPN: https://github.com/foolwood/DaSiamRPN
- benchmark results: https://github.com/foolwood/benchmark_results
- DAT: https://github.com/shipubupt/NIPS2018
- otb-toolkit: https://github.com/ZhouYzzz/otb-toolkit
- SiamRPN: https://github.com/zkisthebest/Siamese-RPN