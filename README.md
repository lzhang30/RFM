# RandFreqMask
A description projection for our JSCAS proposed work 

**"Towards Better Semi-Supervised Multi-Organ Segmentation from CT Volumes: Random Frequency Masking and Pseudo-Label Refinement".**

Run the code for data preprocessing:

```
python code/data/preprocess_acc.py
```

Change the data path in 

```
code/utils/config.py
```


You need to modify the location where the data is stored before the preprocess.

Then, train the model with 5%,10%, and 20% labeled volumes:
```
bash train3times_acc_5_10_20.sh
```

You can also test the weights we pre-trained by

```
tar -xzvf logs.tar.gz
python code/evaluate_Ntimes2.py --task acc_s --exp Task_acc_s_{labeled_ratios}p/{method} --cps AB
```
The logs.tar.gz can be downloaded here: [logs.tar.gz](https://mega.nz/file/gv0TjCKT#IDJ4iLpX-Aru0-LbNcyWOLdegPHQJ5FOQxFZuqrQaGk)

Have fun!
