

```python
import os
from scipy.io import wavfile
from shutil import copyfile
from sphfile import SPHFile
from WaveNetClassifier import WaveNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
```


```python
# Copy folder structure without files

wav_base_dir = 'TRAIN_WAV'

for path, dirname, filename in os.walk('TRAIN'):
    path_array = path.split(os.sep)
    path_array[0] = wav_base_dir
    new_path = '/'.join(path_array)
    try:
        os.makedirs(new_path)
    except FileExistsError:
        # directory already exists
        pass
```


```python
# Transform NISt Sphere Files to WAV and copy them to the TRAIN_WAV directory.

for path, dirname, filenames in os.walk('TRAIN'):
    for filename in filenames:
        if filename.endswith(".WAV"): 
            file_path = os.path.join(path, filename)
            path_array = file_path.split(os.sep)
            path_array[0] = wav_base_dir
            new_path = '/'.join(path_array)
            print(new_path)
            
            sph = SPHFile(file_path)
            sph.write_wav(new_path)
```


```python
fs, data = wavfile.read('TRAIN_WAV/DR3/FGRW0/SA1.WAV')
display(len(data))

fs, data = wavfile.read('TRAIN_WAV/DR3/FJLG0/SA1.WAV')
display(len(data))

fs, data = wavfile.read('TRAIN_WAV/DR6/MSAT1/SA1.WAV')
display(len(data))
```


```python
# Copy the SA1.WAV files with the identifier folders to SA1_UNIFORM directory.

SA1_base_dir = 'SA1_UNIFORM'

for path, dirname, filenames in os.walk('TRAIN_WAV'):
    for filename in filenames:
        if filename.endswith("SA1.WAV"):
            file_path = os.path.join(path, filename)
            path_array = file_path.split(os.sep)
            path_array[0] = SA1_base_dir
            try:
                os.makedirs(path_array[0] + "/" + path_array[-2])
            except FileExistsError:
                # directory already exists
                pass
            new_path = path_array[0] + "/" + path_array[-2] + "/" + filename
            print(new_path)
            copyfile(file_path, new_path)
```


```python
# Iterate over the SA1.WAV files in the SA1_UNIFORM directory, find the shortest
# wav file, then cut all others to the same size.

SA1_base_dir = 'SA1_UNIFORM'
min_length = 100000
wav_dict = {}

for path, dirnames, filenames in os.walk('SA1_UNIFORM'):
    for dirname in dirnames:
        wav_path = SA1_base_dir + "/" + dirname + "/SA1.WAV"
        fs, data = wavfile.read(wav_path)
        wav_dict[dirname] = data
        if len(data) < min_length:
            min_length = len(data)
            
display(wav_dict)
display(min_length)
```


    {'FAEM0': array([ 0, -9,  1, ..., -1, -1,  7], dtype=int16),
     'FAJW0': array([ 4, -5,  0, ..., -4,  1,  1], dtype=int16),
     'FALK0': array([-3,  9,  4, ..., -3, -9,  4], dtype=int16),
     'FALR0': array([-4, -3, -5, ...,  3,  1,  2], dtype=int16),
     'FAPB0': array([1, 0, 6, ..., 5, 5, 5], dtype=int16),
     'FBAS0': array([-6,  0, -3, ...,  2,  0,  1], dtype=int16),
     'FBCG1': array([ 2, -2,  2, ...,  9,  2, 24], dtype=int16),
     'FBCH0': array([ 4,  2,  2, ..., -1,  8, -1], dtype=int16),
     'FBJL0': array([  2,   1,  -1, ...,   1,   8, -20], dtype=int16),
     'FBLV0': array([ 5, 14, 19, ...,  5,  2,  5], dtype=int16),
     'FBMH0': array([ 3, 10, -5, ...,  3, -1,  9], dtype=int16),
     'FBMJ0': array([13, -1, -1, ..., -1,  1, -1], dtype=int16),
     'FCAG0': array([ 2,  0, -2, ...,  2,  2, 15], dtype=int16),
     'FCAJ0': array([ 3, 10, 10, ..., -5, -6, -3], dtype=int16),
     'FCDR1': array([7, 2, 3, ..., 5, 4, 5], dtype=int16),
     'FCEG0': array([ 3,  5,  3, ...,  0,  1, -1], dtype=int16),
     'FCJF0': array([ 1, -1,  2, ..., -1, -5, -8], dtype=int16),
     'FCJS0': array([3, 7, 4, ..., 5, 0, 3], dtype=int16),
     'FCKE0': array([-3,  6,  2, ...,  1,  1, -1], dtype=int16),
     'FCLT0': array([ 4,  1,  1, ..., -4, -4, -2], dtype=int16),
     'FCMG0': array([-23,   5,  -4, ...,   0,   2,   7], dtype=int16),
     'FCMM0': array([23,  9, 17, ...,  4,  3, -1], dtype=int16),
     'FCRZ0': array([ -3, -16,  10, ...,   0,   2,  -3], dtype=int16),
     'FCYL0': array([15, 11, -4, ..., 10,  2, -2], dtype=int16),
     'FDAS1': array([-5, -4, -5, ...,  3,  1,  7], dtype=int16),
     'FDAW0': array([  1,   4,   1, ...,   3,   5, -11], dtype=int16),
     'FDFB0': array([  9,  -5,   0, ...,   9,  11, -19], dtype=int16),
     'FDJH0': array([ 4,  3,  2, ...,  6,  7, -1], dtype=int16),
     'FDKN0': array([-3,  0, -2, ...,  2,  2,  4], dtype=int16),
     'FDML0': array([54, -3, 14, ...,  6,  2, -4], dtype=int16),
     'FDMY0': array([-7,  2, -2, ...,  1,  2,  0], dtype=int16),
     'FDNC0': array([ 9, -7, -2, ..., -1, -3,  5], dtype=int16),
     'FDTD0': array([ 5,  7,  4, ...,  3, -2,  2], dtype=int16),
     'FDXW0': array([ 8,  1,  5, ..., 10, 12,  3], dtype=int16),
     'FEAC0': array([ 1,  3, -6, ..., -5, -1, -4], dtype=int16),
     'FEAR0': array([-2,  0, -2, ...,  2,  2, -1], dtype=int16),
     'FECD0': array([ 3,  1, -1, ...,  1,  2,  4], dtype=int16),
     'FEEH0': array([-1,  9,  0, ...,  0,  1,  2], dtype=int16),
     'FEME0': array([ 2, 10,  8, ..., -3,  8, -1], dtype=int16),
     'FETB0': array([-9,  6, -1, ...,  5,  0,  5], dtype=int16),
     'FEXM0': array([ 14, -10,   1, ...,   0,   0,   1], dtype=int16),
     'FGCS0': array([ 3,  1,  2, ...,  0, -2,  3], dtype=int16),
     'FGDP0': array([-1,  0,  4, ...,  4,  6, -1], dtype=int16),
     'FGMB0': array([-2, 12, -7, ...,  0,  0,  2], dtype=int16),
     'FGRW0': array([-20,   6,  -1, ...,  13,  15,  11], dtype=int16),
     'FHLM0': array([-5, 12, 15, ...,  2,  2,  1], dtype=int16),
     'FHXS0': array([5, 4, 3, ..., 5, 3, 4], dtype=int16),
     'FJDM2': array([ 2,  5, -7, ...,  1,  1, -2], dtype=int16),
     'FJEN0': array([ 6,  6,  6, ...,  2,  2, -2], dtype=int16),
     'FJHK0': array([5, 7, 4, ..., 0, 0, 0], dtype=int16),
     'FJKL0': array([1, 4, 4, ..., 7, 2, 0], dtype=int16),
     'FJLG0': array([-10,   8,  -1, ...,   2,  -1,   1], dtype=int16),
     'FJLR0': array([-5, -3, -4, ..., -3, -1, 10], dtype=int16),
     'FJRB0': array([-10,  10,   7, ...,   3,   3,  -1], dtype=int16),
     'FJRP1': array([13, 11, 11, ...,  1,  3,  0], dtype=int16),
     'FJSK0': array([11, 23, 16, ...,  2, -2,  8], dtype=int16),
     'FJSP0': array([ 8, -3,  2, ...,  2,  4,  0], dtype=int16),
     'FJWB1': array([ 3,  3, 13, ...,  2,  0,  0], dtype=int16),
     'FJXM0': array([ 4, -4,  2, ...,  0,  0,  8], dtype=int16),
     'FJXP0': array([ 5, -2,  1, ...,  1,  1,  0], dtype=int16),
     'FKAA0': array([2, 0, 0, ..., 2, 0, 0], dtype=int16),
     'FKDE0': array([16, -5,  7, ...,  4,  0,  4], dtype=int16),
     'FKDW0': array([ 8, -2,  1, ..., -3, -3,  5], dtype=int16),
     'FKFB0': array([ 2,  2, -1, ..., -2,  0,  7], dtype=int16),
     'FKKH0': array([ -7,  -7, -22, ...,  -2,   3,   2], dtype=int16),
     'FKLC0': array([ 2,  6,  6, ..., -6, -3, -2], dtype=int16),
     'FKLC1': array([ 1,  6,  3, ...,  7, 14, 16], dtype=int16),
     'FKLH0': array([15, 10, 12, ..., -1, -1, -2], dtype=int16),
     'FKSR0': array([ 9, -4,  7, ...,  2,  3,  1], dtype=int16),
     'FLAC0': array([ 7, -3, -8, ..., -2,  7,  4], dtype=int16),
     'FLAG0': array([ 5,  2, -1, ..., -2,  3, -1], dtype=int16),
     'FLEH0': array([2, 6, 3, ..., 6, 7, 2], dtype=int16),
     'FLET0': array([-10,  -2,  -5, ...,  -3,  -4,   2], dtype=int16),
     'FLHD0': array([-10,   2,  -1, ...,  -3,  -1,   6], dtype=int16),
     'FLJA0': array([ 0,  3, -5, ...,  1,  6,  4], dtype=int16),
     'FLJD0': array([2, 0, 2, ..., 0, 5, 0], dtype=int16),
     'FLJG0': array([  3,   2,  -3, ...,  -8, -10,   3], dtype=int16),
     'FLKM0': array([-17,   4,  -2, ...,   5,   2,   2], dtype=int16),
     'FLMA0': array([12, -3,  3, ...,  0,  3,  2], dtype=int16),
     'FLMC0': array([14, -3,  5, ...,  0,  1,  1], dtype=int16),
     'FLMK0': array([4, 6, 1, ..., 4, 3, 1], dtype=int16),
     'FLOD0': array([ 8,  1,  3, ...,  2, -5, -3], dtype=int16),
     'FLTM0': array([0, 4, 4, ..., 1, 4, 4], dtype=int16),
     'FMAH1': array([ 17,  13, -11, ...,  -5,   4,  -2], dtype=int16),
     'FMBG0': array([ 9, -3,  0, ...,  1, -2,  2], dtype=int16),
     'FMEM0': array([-1, -3,  0, ...,  0, -2, -4], dtype=int16),
     'FMJB0': array([-8, -3, -5, ...,  1,  0,  2], dtype=int16),
     'FMJF0': array([-14,  17,  10, ...,   1,  -1,   0], dtype=int16),
     'FMJU0': array([-1,  3, -1, ...,  1, -1, 12], dtype=int16),
     'FMKC0': array([1, 0, 2, ..., 5, 5, 5], dtype=int16),
     'FMKF0': array([ 1, -1,  2, ...,  3,  2, -2], dtype=int16),
     'FMMH0': array([-12, -11,  -4, ...,  12,  15,  12], dtype=int16),
     'FMPG0': array([1, 4, 5, ..., 7, 2, 5], dtype=int16),
     'FNKL0': array([ 9, 11,  5, ...,  2,  3,  3], dtype=int16),
     'FNTB0': array([ 3,  3,  3, ..., -1,  1,  1], dtype=int16),
     'FPAB1': array([-1,  0,  2, ...,  4,  5,  7], dtype=int16),
     'FPAC0': array([-2, -6, -5, ...,  3, -1,  1], dtype=int16),
     'FPAD0': array([ 1, -1, -1, ...,  0,  3,  4], dtype=int16),
     'FPAF0': array([5, 2, 5, ..., 4, 0, 2], dtype=int16),
     'FPAZ0': array([5, 1, 7, ..., 4, 0, 3], dtype=int16),
     'FPJF0': array([1, 8, 6, ..., 5, 4, 6], dtype=int16),
     'FPLS0': array([ 6, -1, -6, ...,  5,  5,  7], dtype=int16),
     'FPMY0': array([  3, -12,  -2, ..., -23, -36, -20], dtype=int16),
     'FREH0': array([-8,  2, -7, ...,  4, -2, -6], dtype=int16),
     'FRJB0': array([-17,   0,  -8, ...,  -1,  -1,   2], dtype=int16),
     'FRLL0': array([-7,  3,  5, ...,  0,  1,  0], dtype=int16),
     'FSAG0': array([ 4, -2, 15, ...,  2,  4, -1], dtype=int16),
     'FSAH0': array([ 4, -2,  0, ...,  2, -3,  0], dtype=int16),
     'FSAK0': array([28,  6, 12, ..., -2, -3, -2], dtype=int16),
     'FSBK0': array([ 2,  1,  3, ..., -2,  0, -3], dtype=int16),
     'FSCN0': array([ 9, -1,  1, ...,  4,  0,  0], dtype=int16),
     'FSDC0': array([-7,  3, -7, ...,  4,  2,  8], dtype=int16),
     'FSDJ0': array([ 6,  4,  3, ...,  1, -1, -1], dtype=int16),
     'FSGF0': array([-3, -6,  1, ...,  4,  5,  4], dtype=int16),
     'FSJG0': array([5, 7, 3, ..., 5, 6, 7], dtype=int16),
     'FSJK1': array([-2,  0,  1, ...,  3,  3,  3], dtype=int16),
     'FSJS0': array([ 3,  0,  0, ...,  0,  2, -2], dtype=int16),
     'FSJW0': array([ 0, -2, -1, ...,  0, -1, -1], dtype=int16),
     'FSKC0': array([14, 16,  9, ..., -5, -7,  6], dtype=int16),
     'FSKL0': array([12, 20, 15, ...,  9,  9, 11], dtype=int16),
     'FSKP0': array([  6,   8,   4, ...,  21,  26, -14], dtype=int16),
     'FSLS0': array([ 6, 13, 22, ...,  3,  2,  4], dtype=int16),
     'FSMA0': array([ 5,  6,  6, ..., -2,  1, -2], dtype=int16),
     'FSMM0': array([13, -2,  4, ..., -1, -1,  2], dtype=int16),
     'FSMS1': array([1, 3, 5, ..., 3, 1, 5], dtype=int16),
     'FSPM0': array([-3,  5, -2, ...,  4,  0,  0], dtype=int16),
     'FSRH0': array([-3, -3,  4, ...,  0, -2,  4], dtype=int16),
     'FSSB0': array([ 4,  3,  3, ...,  0,  4, -4], dtype=int16),
     'FTAJ0': array([-4,  5, -3, ..., 12, 10, 13], dtype=int16),
     'FTBR0': array([13, 11, -6, ...,  3,  6,  6], dtype=int16),
     'FTBW0': array([-6,  5,  1, ..., -2,  0,  2], dtype=int16),
     'FTLG0': array([-18,   1,  -4, ...,  -1,   1,  -2], dtype=int16),
     'FTMG0': array([-5,  3, 10, ...,  1,  2,  1], dtype=int16),
     'FVFB0': array([4, 5, 5, ..., 3, 6, 5], dtype=int16),
     'FVKB0': array([ 5,  0,  0, ..., -5,  2, 10], dtype=int16),
     'FVMH0': array([-1,  4,  1, ...,  0,  1,  2], dtype=int16),
     'MABC0': array([ 12,   5,   0, ...,  -7, -14,  -4], dtype=int16),
     'MADC0': array([-20,   6,  14, ...,   4,   2,  11], dtype=int16),
     'MADD0': array([-12,   3,  -1, ...,   1,  -7,   8], dtype=int16),
     'MAEB0': array([-4, -5, -4, ..., -1,  2, -1], dtype=int16),
     'MAEO0': array([11,  1, -8, ..., -2, -1,  4], dtype=int16),
     'MAFM0': array([2, 7, 3, ..., 3, 3, 0], dtype=int16),
     'MAJP0': array([-10, -11, -12, ...,   5,   3,   2], dtype=int16),
     'MAKB0': array([-1,  1,  2, ..., -1,  2,  7], dtype=int16),
     'MAKR0': array([-7,  1,  1, ...,  2,  2, -2], dtype=int16),
     'MAPV0': array([ 1,  2,  0, ..., -1, 11,  1], dtype=int16),
     'MARC0': array([-6, -1,  2, ...,  5,  4,  5], dtype=int16),
     'MARW0': array([-17,   9,  -1, ...,  -1,   5,  -2], dtype=int16),
     'MBAR0': array([15, -2,  3, ...,  2, -2, -4], dtype=int16),
     'MBBR0': array([ 1,  7,  4, ...,  1,  7, -5], dtype=int16),
     'MBCG0': array([ 4,  1, -2, ...,  3,  1, -3], dtype=int16),
     'MBEF0': array([10, -1,  6, ...,  8,  3, 10], dtype=int16),
     'MBGT0': array([10, 10, 19, ..., -1,  1,  1], dtype=int16),
     'MBJV0': array([-8,  0,  0, ...,  0, -1, -3], dtype=int16),
     'MBMA0': array([-29,   4,  -4, ...,   1,   1,   1], dtype=int16),
     'MBMA1': array([ 2,  5,  3, ...,  6, -1, 12], dtype=int16),
     'MBML0': array([25, 15,  9, ...,  1,  0,  3], dtype=int16),
     'MBOM0': array([ 3,  6, 13, ...,  6,  6,  4], dtype=int16),
     'MBSB0': array([-15,   4,   0, ...,   0,   1,  -4], dtype=int16),
     'MBTH0': array([-7, -6, -4, ...,  3,  0, 10], dtype=int16),
     'MBWP0': array([12,  0,  5, ...,  0,  3,  0], dtype=int16),
     'MCAE0': array([-6,  8,  2, ...,  0,  1, -3], dtype=int16),
     'MCAL0': array([34, -6,  8, ..., -1,  1, -1], dtype=int16),
     'MCDC0': array([ 4, -1,  2, ...,  3,  2,  2], dtype=int16),
     'MCDD0': array([-2,  1,  1, ...,  3, -2, -4], dtype=int16),
     'MCDR0': array([  0,  -2,   0, ..., -12,  -7,  -3], dtype=int16),
     'MCEF0': array([-2,  3,  4, ...,  0, -2,  1], dtype=int16),
     'MCEW0': array([-6, 15,  9, ...,  2, -3, -2], dtype=int16),
     'MCHL0': array([11, 10,  9, ...,  0,  1,  2], dtype=int16),
     'MCLK0': array([ 0, -4, -2, ...,  0, -1, -3], dtype=int16),
     'MCLM0': array([ 4,  7, -4, ...,  0, -2, -2], dtype=int16),
     'MCPM0': array([ 4, -1,  1, ...,  3,  4,  3], dtype=int16),
     'MCRE0': array([-3, -3,  1, ...,  7, 15, -3], dtype=int16),
     'MCSS0': array([-11,   7,  -3, ...,   2,   7,   5], dtype=int16),
     'MCTH0': array([  3,   2, -13, ...,  -2,   0,  -2], dtype=int16),
     'MCTM0': array([-9, 14,  9, ...,  0,  1, -2], dtype=int16),
     'MCXM0': array([20, 21,  4, ...,  5,  1,  6], dtype=int16),
     'MDAC0': array([101, -28,  18, ...,   3,   1,   8], dtype=int16),
     'MDAS0': array([ 2,  1,  5, ..., 11, 14,  5], dtype=int16),
     'MDBB1': array([-2, -1,  0, ...,  6,  7,  5], dtype=int16),
     'MDBP0': array([ 1, -5, -3, ...,  3,  5, -5], dtype=int16),
     'MDCD0': array([  6,  12,   5, ...,   0,   1, -23], dtype=int16),
     'MDCM0': array([ 4,  1,  3, ...,  2,  0, -2], dtype=int16),
     'MDDC0': array([  4, -10, -23, ...,   2,  -1, -14], dtype=int16),
     'MDED0': array([-31,   5, -13, ...,  -1,   0,  -2], dtype=int16),
     'MDEF0': array([ 5, 12, 11, ...,  1,  1, 30], dtype=int16),
     'MDEM0': array([-6,  3, -1, ...,  2, -1,  7], dtype=int16),
     'MDHL0': array([12, -2,  2, ...,  2,  1,  6], dtype=int16),
     'MDHS0': array([ 0,  2,  1, ..., -6, -8,  8], dtype=int16),
     'MDJM0': array([ 12, -10,  -7, ...,   2,  -3,  -2], dtype=int16),
     'MDKS0': array([-6,  5, 10, ...,  6,  0, 16], dtype=int16),
     'MDLB0': array([-1, -2, -3, ...,  4,  2,  2], dtype=int16),
     'MDLC0': array([0, 1, 0, ..., 2, 2, 5], dtype=int16),
     'MDLC1': array([2, 3, 3, ..., 0, 1, 3], dtype=int16),
     'MDLC2': array([ 5, -1,  0, ...,  5,  2,  1], dtype=int16),
     'MDLH0': array([ 1, -2,  0, ..., -6, -3,  1], dtype=int16),
     'MDLM0': array([-5,  2, -1, ...,  8, -3,  4], dtype=int16),
     'MDLR0': array([ -9,   7,   4, ...,   0,   0, -15], dtype=int16),
     'MDLR1': array([-7,  6,  0, ..., -2,  0, -3], dtype=int16),
     'MDMA0': array([-25,   9,  -1, ...,  -3,  -7,  -1], dtype=int16),
     'MDMT0': array([10, 12,  9, ...,  3,  2, 11], dtype=int16),
     'MDNS0': array([-3, -2,  3, ...,  0,  4,  1], dtype=int16),
     'MDPB0': array([ 8,  9,  9, ...,  7,  6, -5], dtype=int16),
     'MDPK0': array([3, 5, 1, ..., 6, 7, 4], dtype=int16),
     'MDPS0': array([ 9,  1,  1, ..., -2,  2,  0], dtype=int16),
     'MDRD0': array([26,  7,  8, ...,  1,  3,  3], dtype=int16),
     'MDSJ0': array([18,  7,  6, ..., -8, -1, -4], dtype=int16),
     'MDSS0': array([5, 1, 1, ..., 2, 4, 2], dtype=int16),
     'MDSS1': array([ 0,  1,  0, ...,  8,  8, 13], dtype=int16),
     'MDTB0': array([-3,  3, -1, ..., -3, -6,  9], dtype=int16),
     'MDWD0': array([-8,  6, 12, ...,  1,  5, 12], dtype=int16),
     'MDWH0': array([ 9,  0,  1, ..., -2,  2,  0], dtype=int16),
     'MDWM0': array([ 0,  4,  4, ..., -3,  0, -8], dtype=int16),
     'MEAL0': array([5, 4, 6, ..., 5, 3, 3], dtype=int16),
     'MEDR0': array([-2,  5,  2, ...,  3,  3,  3], dtype=int16),
     'MEFG0': array([ 39, -11,   4, ...,   7,   6,  22], dtype=int16),
     'MEGJ0': array([7, 1, 2, ..., 0, 2, 1], dtype=int16),
     'MEJL0': array([1, 2, 5, ..., 1, 2, 5], dtype=int16),
     'MEJS0': array([-6, -1,  0, ...,  3,  1,  4], dtype=int16),
     'MESG0': array([14, -4,  4, ..., -3, -2, -8], dtype=int16),
     'MESJ0': array([2, 3, 4, ..., 3, 4, 0], dtype=int16),
     'MEWM0': array([13, -3,  6, ...,  8,  2, 12], dtype=int16),
     'MFER0': array([ 3,  1,  2, ...,  1, -1,  3], dtype=int16),
     'MFMC0': array([ 8, -1, 14, ...,  2,  0, -1], dtype=int16),
     'MFRM0': array([-4, -5,  2, ...,  3,  0, -1], dtype=int16),
     'MFWK0': array([-15,   5,  -2, ...,  -3,  -3,  -2], dtype=int16),
     'MFXS0': array([3, 0, 2, ..., 5, 5, 4], dtype=int16),
     'MFXV0': array([-1, -3, -6, ...,  1, -1,  2], dtype=int16),
     'MGAF0': array([-19,   8,  -2, ...,   1,  -2,   7], dtype=int16),
     'MGAG0': array([-11,   7,  -4, ...,   2,  -2,   9], dtype=int16),
     'MGAK0': array([4, 4, 3, ..., 6, 3, 4], dtype=int16),
     'MGAR0': array([-6, -3, -3, ..., -5,  4, -4], dtype=int16),
     'MGAW0': array([14, -6, -1, ..., -1,  1, -1], dtype=int16),
     'MGES0': array([-1,  1,  0, ..., 10, 12,  1], dtype=int16),
     'MGJC0': array([ 8, -5,  1, ..., -2, -1, -5], dtype=int16),
     'MGRL0': array([13, 18,  9, ..., -5, -3, -4], dtype=int16),
     'MGRP0': array([-7,  1,  7, ...,  0, -1, -2], dtype=int16),
     'MGSH0': array([ -1, -16,   2, ...,  -1,  -4,  -6], dtype=int16),
     'MGSL0': array([ 7,  7,  3, ..., 10, 12,  8], dtype=int16),
     'MGXP0': array([10, -1,  5, ...,  5,  2, -3], dtype=int16),
     'MHBS0': array([ 3,  2,  4, ..., -1,  2, -4], dtype=int16),
     'MHIT0': array([1, 6, 1, ..., 3, 7, 5], dtype=int16),
     'MHJB0': array([-3, -2,  3, ...,  2,  0,  4], dtype=int16),
     'MHMG0': array([-12,   4,   1, ...,   6,   1,   2], dtype=int16),
     'MHMR0': array([-5,  5,  2, ...,  0,  6, -9], dtype=int16),
     'MHRM0': array([-10,   1,  15, ...,   2,   6,  -5], dtype=int16),
     'MHXL0': array([  5,  -3,   1, ...,  -5,  -1, -20], dtype=int16),
     'MILB0': array([-11,  -8,  -7, ...,  -6,  11,  -6], dtype=int16),
     'MJAC0': array([11, -2,  3, ...,  6,  7, -1], dtype=int16),
     'MJAE0': array([-2,  7, 10, ..., 15,  7, 30], dtype=int16),
     'MJAI0': array([3, 5, 2, ..., 2, 0, 2], dtype=int16),
     'MJBG0': array([-8,  3, -1, ...,  0, -4, -4], dtype=int16),
     'MJDA0': array([ 2,  3,  2, ...,  4,  1, -3], dtype=int16),
     'MJDC0': array([8, 0, 3, ..., 0, 0, 3], dtype=int16),
     'MJDE0': array([14, -1,  3, ...,  2,  0,  4], dtype=int16),
     'MJDG0': array([0, 5, 5, ..., 4, 1, 2], dtype=int16),
     'MJDM0': array([-9,  5,  3, ...,  0,  2, -2], dtype=int16),
     'MJEB0': array([10,  0, -3, ..., -5, -3,  0], dtype=int16),
     'MJEB1': array([ 6,  2, -4, ..., -1,  2,  3], dtype=int16),
     'MJEE0': array([-4,  5,  1, ...,  3, -4,  2], dtype=int16),
     'MJFH0': array([-2,  7,  1, ..., -2,  4,  1], dtype=int16),
     'MJFR0': array([ 5, -1,  0, ...,  3,  7, -5], dtype=int16),
     'MJHI0': array([-9,  4,  2, ...,  2,  5, -4], dtype=int16),
     'MJJB0': array([ 16,   9,   6, ...,  -1,   6, -13], dtype=int16),
     'MJJJ0': array([12, -4,  0, ...,  8, -1,  0], dtype=int16),
     'MJJM0': array([1, 3, 1, ..., 2, 3, 2], dtype=int16),
     'MJKR0': array([-9,  1, -2, ..., -3, -6, -4], dtype=int16),
     'MJLB0': array([ 0,  4,  1, ..., -7, -3,  0], dtype=int16),
     'MJLG1': array([4, 4, 3, ..., 7, 1, 0], dtype=int16),
     'MJLS0': array([ 11, -18, -14, ...,   2,   3,   0], dtype=int16),
     'MJMA0': array([-1,  2,  2, ..., -1,  2, -6], dtype=int16),
     'MJMD0': array([4, 6, 6, ..., 6, 2, 6], dtype=int16),
     'MJMM0': array([-8,  1,  0, ..., -3, -3, -3], dtype=int16),
     'MJPG0': array([-12,   5,  -1, ...,   1,  -1,   2], dtype=int16),
     'MJPM0': array([-5,  5,  1, ..., -3, -5,  2], dtype=int16),
     'MJPM1': array([-4, -2,  4, ...,  4,  1, 12], dtype=int16),
     'MJRA0': array([20,  6, 11, ..., -1, -2,  1], dtype=int16),
     'MJRG0': array([ 7,  1,  5, ..., -2, -4, -4], dtype=int16),
     'MJRH0': array([13, -4,  2, ...,  2,  4,  1], dtype=int16),
     'MJRH1': array([-4,  5,  2, ...,  3, -1,  0], dtype=int16),
     'MJRK0': array([ 13,  14,   3, ...,  -3,   1, -15], dtype=int16),
     'MJRP0': array([24,  7,  4, ..., -3, -5, -2], dtype=int16),
     'MJSR0': array([ 0,  5,  1, ...,  3, -1,  0], dtype=int16),
     'MJWG0': array([ 9,  3, -7, ...,  1,  2, -1], dtype=int16),
     'MJWS0': array([ 6, -1,  2, ...,  3,  0,  0], dtype=int16),
     'MJWT0': array([-6,  2,  1, ..., -1, -1, -4], dtype=int16),
     'MJXA0': array([-8, -7,  6, ...,  4,  1, -2], dtype=int16),
     'MJXL0': array([15,  0, -6, ...,  2, -3, -1], dtype=int16),
     'MKAG0': array([3, 2, 3, ..., 2, 3, 1], dtype=int16),
     'MKAH0': array([11, 11, 16, ..., -1,  4, -6], dtype=int16),
     'MKAJ0': array([17, -3,  1, ...,  1, -1,  7], dtype=int16),
     'MKAM0': array([-8,  8, 11, ..., -2, -1,  2], dtype=int16),
     'MKDB0': array([ 0, -1, -1, ...,  0, -1, -3], dtype=int16),
     'MKDD0': array([ 8,  2, -9, ..., -4, -1, -5], dtype=int16),
     'MKDT0': array([-2,  2,  0, ..., -1,  0, -1], dtype=int16),
     'MKES0': array([ 5, -2,  1, ...,  2,  2, -3], dtype=int16),
     'MKJO0': array([1, 0, 0, ..., 0, 2, 6], dtype=int16),
     'MKLN0': array([3, 6, 5, ..., 2, 3, 3], dtype=int16),
     'MKLR0': array([ -5,   3,   3, ..., -18, -19, -17], dtype=int16),
     'MKLS0': array([ 9, 15,  4, ...,  1,  1,  1], dtype=int16),
     'MKLS1': array([  2,   0,   5, ...,  -1,   4, -10], dtype=int16),
     'MKLW0': array([ 1,  0,  3, ..., -1,  0,  1], dtype=int16),
     'MKRG0': array([ 3, 10,  5, ...,  2,  2, -1], dtype=int16),
     'MKXL0': array([ 9, -2,  3, ..., 31, 25,  6], dtype=int16),
     'MLBC0': array([ 4, 13, 11, ..., -2,  2, -9], dtype=int16),
     'MLEL0': array([ 5, -1,  1, ..., -4, -5,  0], dtype=int16),
     'MLJC0': array([-9,  2, -1, ...,  0,  2, -1], dtype=int16),
     'MLJH0': array([20, -6,  3, ..., -4,  4,  7], dtype=int16),
     'MLNS0': array([ 2, 10, -4, ...,  5,  2,  2], dtype=int16),
     'MLSH0': array([ 31, -10,   2, ...,   3,   2,   6], dtype=int16),
     'MMAA0': array([-5, -5, -2, ..., -1,  1,  3], dtype=int16),
     'MMAB1': array([ 0,  3,  0, ..., -1,  4, -4], dtype=int16),
     'MMAG0': array([-16, -10, -10, ...,  -3,   1,   0], dtype=int16),
     'MMAM0': array([0, 5, 5, ..., 4, 0, 1], dtype=int16),
     'MMAR0': array([ 5, -3,  9, ...,  1,  1, 13], dtype=int16),
     'MMBS0': array([ 6,  0,  1, ..., 12,  3, 32], dtype=int16),
     'MMCC0': array([-19,   2,   3, ...,  -1,   2,   2], dtype=int16),
     'MMDB0': array([ 5,  3,  1, ...,  2, -2,  3], dtype=int16),
     'MMDG0': array([-7,  7,  1, ...,  3,  4,  8], dtype=int16),
     'MMDM0': array([ 6, -5, -1, ...,  1,  2,  1], dtype=int16),
     'MMDM1': array([1, 4, 4, ..., 0, 1, 0], dtype=int16),
     'MMDS0': array([ 6,  8, 13, ...,  0, -3, -1], dtype=int16),
     'MMEA0': array([ 6,  3,  0, ...,  0,  0, -5], dtype=int16),
     'MMEB0': array([10, -9,  0, ...,  4,  2,  5], dtype=int16),
     'MMGC0': array([ 2,  1,  4, ...,  2, -2,  2], dtype=int16),
     'MMGG0': array([-3,  7,  6, ...,  3,  3, -1], dtype=int16),
     'MMGK0': array([-6,  3,  1, ..., -4, -5, -5], dtype=int16),
     'MMJB1': array([-5,  5,  1, ...,  9, 10,  6], dtype=int16),
     'MMLM0': array([ 1,  0,  2, ..., -3, -3,  9], dtype=int16),
     'MMPM0': array([15, 23, 20, ...,  6,  5,  3], dtype=int16),
     'MMRP0': array([11,  0,  4, ...,  0,  0, -2], dtype=int16),
     'MMSM0': array([11, -5,  4, ...,  1,  3,  4], dtype=int16),
     'MMVP0': array([-8,  4,  1, ...,  2,  0, -1], dtype=int16),
     'MMWB0': array([6, 6, 6, ..., 2, 5, 5], dtype=int16),
     'MMWS0': array([-2,  1,  2, ..., 27, 17, -5], dtype=int16),
     'MMWS1': array([ 1, -2,  3, ...,  3,  5,  6], dtype=int16),
     'MMXS0': array([ 0,  2,  0, ..., -1,  1,  3], dtype=int16),
     'MNET0': array([-9,  3,  0, ..., -1, -1, -1], dtype=int16),
     'MNTW0': array([-5, -4,  4, ...,  4,  4,  3], dtype=int16),
     'MPAR0': array([ 1,  4,  4, ..., 18, -7, 36], dtype=int16),
     'MPEB0': array([-11,   4,  -1, ...,   0,   0,   4], dtype=int16),
     'MPFU0': array([-13, -28, -10, ...,   4,  -2,   3], dtype=int16),
     'MPGH0': array([2, 7, 5, ..., 0, 0, 6], dtype=int16),
     'MPGR0': array([ 8,  1,  7, ...,  7, -4,  6], dtype=int16),
     'MPGR1': array([-6, -3,  2, ...,  1,  2,  1], dtype=int16),
     'MPMB0': array([1, 0, 0, ..., 1, 3, 0], dtype=int16),
     'MPPC0': array([ 0, 18, 19, ...,  1,  9,  2], dtype=int16),
     'MPRB0': array([ 4, -2, -1, ..., -4, -2,  0], dtype=int16),
     'MPRD0': array([ 23,   4,   7, ...,   8,   2, -12], dtype=int16),
     'MPRK0': array([-6,  1, -1, ...,  0, -1, -1], dtype=int16),
     'MPRT0': array([11, -8, -3, ...,  6,  2, -5], dtype=int16),
     'MPSW0': array([5, 2, 1, ..., 1, 4, 6], dtype=int16),
     'MRAB0': array([-7, 14,  2, ...,  2, 11,  6], dtype=int16),
     'MRAB1': array([ 2, -2, -2, ...,  1,  1,  0], dtype=int16),
     'MRAI0': array([-4,  2,  0, ...,  0,  1,  1], dtype=int16),
     'MRAM0': array([-10,   4,  -3, ...,   4,  10,  10], dtype=int16),
     'MRAV0': array([ 0,  0, -1, ...,  4,  6,  6], dtype=int16),
     'MRBC0': array([ 1,  5,  4, ..., -6, -5, -7], dtype=int16),
     'MRCG0': array([11,  7,  5, ...,  4, -3, -4], dtype=int16),
     'MRCW0': array([-1, 13,  6, ...,  1, -2,  0], dtype=int16),
     'MRDD0': array([1, 4, 2, ..., 3, 2, 1], dtype=int16),
     'MRDM0': array([-16, -18, -12, ...,  -1,   1,  -2], dtype=int16),
     'MRDS0': array([-9,  3, -1, ...,  3,  1, -1], dtype=int16),
     'MREE0': array([  9,   6,  15, ...,  -7,  -8, -13], dtype=int16),
     'MREH1': array([ 3,  3, -1, ...,  3, -1,  1], dtype=int16),
     'MREM0': array([5, 6, 1, ..., 6, 6, 9], dtype=int16),
     'MREW1': array([-2,  0,  1, ...,  1,  1,  4], dtype=int16),
     'MRFK0': array([ 3,  0, -2, ...,  3,  7, -4], dtype=int16),
     'MRFL0': array([-18, -14, -15, ...,   8,  11,   6], dtype=int16),
     'MRGM0': array([-38,  10,  -7, ...,   1,  -1,  -1], dtype=int16),
     'MRGS0': array([  3,   3,   3, ...,  -6,  -2, -28], dtype=int16),
     'MRHL0': array([0, 4, 3, ..., 1, 2, 2], dtype=int16),
     'MRJB1': array([5, 7, 3, ..., 7, 6, 7], dtype=int16),
     'MRJH0': array([2, 2, 2, ..., 3, 6, 1], dtype=int16),
     'MRJM0': array([ 4, -2,  0, ...,  4,  1,  1], dtype=int16),
     'MRJM1': array([ 1, -1,  0, ...,  1,  0,  0], dtype=int16),
     'MRJT0': array([11, 13,  2, ...,  1,  0,  4], dtype=int16),
     'MRKM0': array([-7,  4, -2, ...,  4, -3, 23], dtype=int16),
     'MRLD0': array([5, 7, 4, ..., 1, 0, 5], dtype=int16),
     'MRLJ0': array([-5,  4,  2, ..., 11, 10,  8], dtype=int16),
     'MRLJ1': array([ 2,  3,  3, ..., -1, -1,  0], dtype=int16),
     'MRLK0': array([  3,   2,  -1, ...,  -8,   2, -32], dtype=int16),
     'MRLR0': array([21, 12, 14, ..., -4, -3, -5], dtype=int16),
     'MRMB0': array([1, 1, 5, ..., 6, 5, 7], dtype=int16),
     'MRMG0': array([ 3,  6, 10, ...,  3,  4,  3], dtype=int16),
     'MRMH0': array([ 10,  -2, -21, ...,   2,   3,   1], dtype=int16),
     'MRML0': array([ -6,  -4,  -1, ..., -12,   1,   9], dtype=int16),
     'MRMS0': array([ -5, -15, -10, ...,   3,   2,   9], dtype=int16),
     'MRPC1': array([ 0,  1,  4, ...,  3, -1, 12], dtype=int16),
     'MRRE0': array([-10,   4,   1, ...,   5,   7,  -3], dtype=int16),
     'MRSO0': array([5, 6, 6, ..., 1, 0, 1], dtype=int16),
     'MRSP0': array([13, 10, 12, ..., 12,  9,  7], dtype=int16),
     'MRTC0': array([-9,  6,  2, ...,  4,  2,  1], dtype=int16),
     'MRTJ0': array([-6,  6,  3, ...,  0,  1, -1], dtype=int16),
     'MRVG0': array([-13,   0,  -4, ...,   8,   3,   0], dtype=int16),
     'MRWA0': array([4, 5, 4, ..., 2, 1, 7], dtype=int16),
     'MRWS0': array([-10,   8,   1, ...,   3,   1,   7], dtype=int16),
     'MRXB0': array([5, 5, 5, ..., 4, 4, 3], dtype=int16),
     'MSAH1': array([ 2,  6,  1, ...,  0, -5, -2], dtype=int16),
     'MSAS0': array([ 3,  0,  1, ...,  1, -1,  3], dtype=int16),
     'MSAT0': array([  1,  -2,  -1, ...,  -6, -10,  -9], dtype=int16),
     'MSAT1': array([5, 5, 5, ..., 2, 0, 1], dtype=int16),
     'MSDB0': array([ 2,  2,  2, ...,  1,  0, -5], dtype=int16),
     'MSDH0': array([0, 2, 2, ..., 3, 2, 6], dtype=int16),
     'MSDS0': array([-2, 10, 11, ...,  8,  8, 10], dtype=int16),
     'MSEM1': array([ -6, -14,  -8, ...,  -3,   2,   0], dtype=int16),
     'MSES0': array([-2,  1, -1, ..., -4, -6, -6], dtype=int16),
     'MSFH0': array([ 0, -1, -1, ...,  0, -2,  0], dtype=int16),
     'MSFV0': array([-3,  1, -7, ...,  1,  1,  0], dtype=int16),
     'MSJK0': array([ 6,  0, -7, ...,  5,  5, -3], dtype=int16),
     'MSMC0': array([-2,  3, -1, ..., -3, -1,  5], dtype=int16),
     'MSMR0': array([-7,  6,  2, ...,  3,  0,  1], dtype=int16),
     'MSMS0': array([-13,   1,   7, ...,  -1,   0,  -7], dtype=int16),
     'MSRG0': array([5, 0, 0, ..., 3, 5, 4], dtype=int16),
     'MSRR0': array([ 7, -3,  0, ..., -3,  3,  7], dtype=int16),
     'MSTF0': array([ 8,  9, 10, ...,  0, -1, -2], dtype=int16),
     'MSVS0': array([2, 1, 1, ..., 4, 7, 4], dtype=int16),
     'MTAB0': array([ 6,  2, -2, ...,  1,  4,  5], dtype=int16),
     'MTAS0': array([-11,   6,  -4, ...,   3,   3,   4], dtype=int16),
     'MTAT0': array([13,  1,  3, ..., -3, -1, -4], dtype=int16),
     'MTAT1': array([-5, 20,  4, ...,  3,  4, -2], dtype=int16),
     'MTBC0': array([-15,   9,   2, ...,  29,  33,  31], dtype=int16),
     'MTCS0': array([-6,  2,  2, ..., -3,  5,  9], dtype=int16),
     'MTDB0': array([ 5,  1,  3, ..., -2,  1, -1], dtype=int16),
     'MTDP0': array([ 0,  3, -1, ...,  1,  1,  0], dtype=int16),
     'MTER0': array([21, -7,  4, ..., 13,  5,  4], dtype=int16),
     'MTJG0': array([2, 4, 5, ..., 6, 6, 4], dtype=int16),
     'MTJM0': array([-8, -1, -1, ...,  8, -4, -4], dtype=int16),
     'MTJS0': array([-15,   5,  -3, ...,   6,  -2,  -2], dtype=int16),
     'MTJU0': array([-3, -3, -1, ...,  4, -2,  1], dtype=int16),
     'MTKD0': array([-2,  2,  0, ...,  2, 15, -1], dtype=int16),
     'MTKP0': array([ 2,  4,  2, ..., -2,  2, -3], dtype=int16),
     'MTLB0': array([ 5, -2,  3, ...,  1,  1, -1], dtype=int16),
     'MTLC0': array([-1,  0,  2, ...,  2, -6,  6], dtype=int16),
     'MTML0': array([  1,   5,   5, ..., -12,   5,  16], dtype=int16),
     'MTMN0': array([5, 4, 3, ..., 4, 2, 4], dtype=int16),
     'MTMT0': array([27, 16, 13, ..., 17, 15, 33], dtype=int16),
     'MTPF0': array([  8,  -4,   3, ...,   1,   7, -13], dtype=int16),
     'MTPG0': array([12, 10,  5, ...,  2,  3,  2], dtype=int16),
     'MTPP0': array([ 9, 10,  7, ..., -1, -1,  0], dtype=int16),
     'MTPR0': array([ 7, 10,  1, ..., -1, -3,  0], dtype=int16),
     'MTQC0': array([  6,  -3,   3, ...,   2, -13,  -9], dtype=int16),
     'MTRC0': array([ 4,  4,  4, ...,  0, -2,  8], dtype=int16),
     'MTRR0': array([5, 2, 3, ..., 6, 6, 3], dtype=int16),
     'MTRT0': array([-10,   5,  -2, ...,   1,  -1,  -6], dtype=int16),
     'MTWH1': array([ 1,  1,  0, ..., -3,  0, -4], dtype=int16),
     'MTXS0': array([ 2,  6,  3, ...,  5, -3,  1], dtype=int16),
     'MVJH0': array([ 1,  1,  0, ..., -2,  0, -1], dtype=int16),
     'MVLO0': array([10,  0, -6, ...,  4, -3, -3], dtype=int16),
     'MVRW0': array([ 3,  4,  4, ...,  3,  5, -2], dtype=int16),
     'MWAC0': array([ 0, -2, -2, ...,  6,  8,  8], dtype=int16),
     'MWAD0': array([  3,  -2, -11, ...,  11,   9,   6], dtype=int16),
     'MWAR0': array([5, 8, 0, ..., 1, 1, 5], dtype=int16),
     'MWCH0': array([1, 4, 3, ..., 7, 0, 6], dtype=int16),
     'MWDK0': array([-5, 12, -5, ...,  4,  0,  2], dtype=int16),
     'MWEM0': array([ 18,   3,  -2, ...,  -7, -15, -12], dtype=int16),
     'MWGR0': array([-8, -9, -3, ..., -1,  0,  5], dtype=int16),
     'MWRE0': array([24, 22, 15, ...,  2, -1, -2], dtype=int16),
     'MWRP0': array([2, 1, 1, ..., 0, 3, 3], dtype=int16),
     'MWSB0': array([-7, -2,  2, ..., -2, -2, -2], dtype=int16),
     'MWSH0': array([-16,   6,   4, ...,  -4,   3, -31], dtype=int16),
     'MZMB0': array([-6,  1, -7, ...,  4,  0,  1], dtype=int16)}



    37581



```python
# Slice arrays for the minimum length

for key in wav_dict:
    wav_dict[key] = wav_dict[key][0 : min_length]
```


```python
# First ten samples to test.

wav_dict_keys_1_10 = list(wav_dict.keys())[0:10]
wav_dict_values_1_10 = list(wav_dict.values())[0:10]

wav_dict_1_10 = dict(zip(wav_dict_keys_1_10, wav_dict_values_1_10))
wav_dict_1_10
```




    {'FAEM0': array([   0,   -9,    1, ..., -423, -476, -502], dtype=int16),
     'FAJW0': array([  4,  -5,   0, ..., -46,  83, -93], dtype=int16),
     'FALK0': array([  -3,    9,    4, ..., -408, -642, -765], dtype=int16),
     'FALR0': array([  -4,   -3,   -5, ..., -248, -215, -165], dtype=int16),
     'FAPB0': array([ 1,  0,  6, ..., 47, 64, 87], dtype=int16),
     'FBAS0': array([ -6,   0,  -3, ..., 164, 180, 165], dtype=int16),
     'FBCG1': array([  2,  -2,   2, ..., 459, 448, 389], dtype=int16),
     'FBCH0': array([   4,    2,    2, ...,  319,  477, -813], dtype=int16),
     'FBJL0': array([   2,    1,   -1, ...,  134, -607, -111], dtype=int16),
     'FBLV0': array([   5,   14,   19, ..., 1860, 1512, 1063], dtype=int16)}




```python
X_train = np.array(wav_dict_values_1_10)
print(X_train.shape)

onehot_encoder = OneHotEncoder(sparse = False, categories = 'auto')
y_train = onehot_encoder.fit_transform(np.array(wav_dict_keys_1_10).reshape(-1, 1))
print(onehot_encoded)
```

    (10, 37581)
    [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
    


```python

wnc = WaveNetClassifier((37581,), (10,), kernel_size = 2, dilation_depth = 9, n_filters = 40, task = 'classification')

#y_pred = wnc.predict(X_test)
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    original_input (InputLayer)     (None, 37581)        0                                            
    __________________________________________________________________________________________________
    reshaped_input (Reshape)        (None, 37581, 1)     0           original_input[0][0]             
    __________________________________________________________________________________________________
    dilated_conv_1 (Conv1D)         (None, 37581, 40)    120         reshaped_input[0][0]             
    __________________________________________________________________________________________________
    dilated_conv_2_tanh (Conv1D)    (None, 37581, 40)    3240        dilated_conv_1[0][0]             
    __________________________________________________________________________________________________
    dilated_conv_2_sigm (Conv1D)    (None, 37581, 40)    3240        dilated_conv_1[0][0]             
    __________________________________________________________________________________________________
    gated_activation_1 (Multiply)   (None, 37581, 40)    0           dilated_conv_2_tanh[0][0]        
                                                                     dilated_conv_2_sigm[0][0]        
    __________________________________________________________________________________________________
    skip_1 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_1[0][0]         
    __________________________________________________________________________________________________
    residual_block_1 (Add)          (None, 37581, 40)    0           skip_1[0][0]                     
                                                                     dilated_conv_1[0][0]             
    __________________________________________________________________________________________________
    dilated_conv_4_tanh (Conv1D)    (None, 37581, 40)    3240        residual_block_1[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_4_sigm (Conv1D)    (None, 37581, 40)    3240        residual_block_1[0][0]           
    __________________________________________________________________________________________________
    gated_activation_2 (Multiply)   (None, 37581, 40)    0           dilated_conv_4_tanh[0][0]        
                                                                     dilated_conv_4_sigm[0][0]        
    __________________________________________________________________________________________________
    skip_2 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_2[0][0]         
    __________________________________________________________________________________________________
    residual_block_2 (Add)          (None, 37581, 40)    0           skip_2[0][0]                     
                                                                     residual_block_1[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_8_tanh (Conv1D)    (None, 37581, 40)    3240        residual_block_2[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_8_sigm (Conv1D)    (None, 37581, 40)    3240        residual_block_2[0][0]           
    __________________________________________________________________________________________________
    gated_activation_3 (Multiply)   (None, 37581, 40)    0           dilated_conv_8_tanh[0][0]        
                                                                     dilated_conv_8_sigm[0][0]        
    __________________________________________________________________________________________________
    skip_3 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_3[0][0]         
    __________________________________________________________________________________________________
    residual_block_3 (Add)          (None, 37581, 40)    0           skip_3[0][0]                     
                                                                     residual_block_2[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_16_tanh (Conv1D)   (None, 37581, 40)    3240        residual_block_3[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_16_sigm (Conv1D)   (None, 37581, 40)    3240        residual_block_3[0][0]           
    __________________________________________________________________________________________________
    gated_activation_4 (Multiply)   (None, 37581, 40)    0           dilated_conv_16_tanh[0][0]       
                                                                     dilated_conv_16_sigm[0][0]       
    __________________________________________________________________________________________________
    skip_4 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_4[0][0]         
    __________________________________________________________________________________________________
    residual_block_4 (Add)          (None, 37581, 40)    0           skip_4[0][0]                     
                                                                     residual_block_3[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_32_tanh (Conv1D)   (None, 37581, 40)    3240        residual_block_4[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_32_sigm (Conv1D)   (None, 37581, 40)    3240        residual_block_4[0][0]           
    __________________________________________________________________________________________________
    gated_activation_5 (Multiply)   (None, 37581, 40)    0           dilated_conv_32_tanh[0][0]       
                                                                     dilated_conv_32_sigm[0][0]       
    __________________________________________________________________________________________________
    skip_5 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_5[0][0]         
    __________________________________________________________________________________________________
    residual_block_5 (Add)          (None, 37581, 40)    0           skip_5[0][0]                     
                                                                     residual_block_4[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_64_tanh (Conv1D)   (None, 37581, 40)    3240        residual_block_5[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_64_sigm (Conv1D)   (None, 37581, 40)    3240        residual_block_5[0][0]           
    __________________________________________________________________________________________________
    gated_activation_6 (Multiply)   (None, 37581, 40)    0           dilated_conv_64_tanh[0][0]       
                                                                     dilated_conv_64_sigm[0][0]       
    __________________________________________________________________________________________________
    skip_6 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_6[0][0]         
    __________________________________________________________________________________________________
    residual_block_6 (Add)          (None, 37581, 40)    0           skip_6[0][0]                     
                                                                     residual_block_5[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_128_tanh (Conv1D)  (None, 37581, 40)    3240        residual_block_6[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_128_sigm (Conv1D)  (None, 37581, 40)    3240        residual_block_6[0][0]           
    __________________________________________________________________________________________________
    gated_activation_7 (Multiply)   (None, 37581, 40)    0           dilated_conv_128_tanh[0][0]      
                                                                     dilated_conv_128_sigm[0][0]      
    __________________________________________________________________________________________________
    skip_7 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_7[0][0]         
    __________________________________________________________________________________________________
    residual_block_7 (Add)          (None, 37581, 40)    0           skip_7[0][0]                     
                                                                     residual_block_6[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_256_tanh (Conv1D)  (None, 37581, 40)    3240        residual_block_7[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_256_sigm (Conv1D)  (None, 37581, 40)    3240        residual_block_7[0][0]           
    __________________________________________________________________________________________________
    gated_activation_8 (Multiply)   (None, 37581, 40)    0           dilated_conv_256_tanh[0][0]      
                                                                     dilated_conv_256_sigm[0][0]      
    __________________________________________________________________________________________________
    skip_8 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_8[0][0]         
    __________________________________________________________________________________________________
    residual_block_8 (Add)          (None, 37581, 40)    0           skip_8[0][0]                     
                                                                     residual_block_7[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_512_tanh (Conv1D)  (None, 37581, 40)    3240        residual_block_8[0][0]           
    __________________________________________________________________________________________________
    dilated_conv_512_sigm (Conv1D)  (None, 37581, 40)    3240        residual_block_8[0][0]           
    __________________________________________________________________________________________________
    gated_activation_9 (Multiply)   (None, 37581, 40)    0           dilated_conv_512_tanh[0][0]      
                                                                     dilated_conv_512_sigm[0][0]      
    __________________________________________________________________________________________________
    skip_9 (Conv1D)                 (None, 37581, 40)    1640        gated_activation_9[0][0]         
    __________________________________________________________________________________________________
    skip_connections (Add)          (None, 37581, 40)    0           skip_1[0][0]                     
                                                                     skip_2[0][0]                     
                                                                     skip_3[0][0]                     
                                                                     skip_4[0][0]                     
                                                                     skip_5[0][0]                     
                                                                     skip_6[0][0]                     
                                                                     skip_7[0][0]                     
                                                                     skip_8[0][0]                     
                                                                     skip_9[0][0]                     
    __________________________________________________________________________________________________
    activation_5 (Activation)       (None, 37581, 40)    0           skip_connections[0][0]           
    __________________________________________________________________________________________________
    conv_5ms (Conv1D)               (None, 37581, 40)    128040      activation_5[0][0]               
    __________________________________________________________________________________________________
    downsample_to_200Hz (AveragePoo (None, 470, 40)      0           conv_5ms[0][0]                   
    __________________________________________________________________________________________________
    conv_500ms (Conv1D)             (None, 470, 40)      160040      downsample_to_200Hz[0][0]        
    __________________________________________________________________________________________________
    conv_500ms_target_shape (Conv1D (None, 470, 10)      40010       conv_500ms[0][0]                 
    __________________________________________________________________________________________________
    downsample_to_2Hz (AveragePooli (None, 5, 10)        0           conv_500ms_target_shape[0][0]    
    __________________________________________________________________________________________________
    final_conv (Conv1D)             (None, 5, 10)        410         downsample_to_2Hz[0][0]          
    __________________________________________________________________________________________________
    final_pooling (AveragePooling1D (None, 1, 10)        0           final_conv[0][0]                 
    __________________________________________________________________________________________________
    reshape_3 (Reshape)             (None, 10)           0           final_pooling[0][0]              
    __________________________________________________________________________________________________
    activation_6 (Activation)       (None, 10)           0           reshape_3[0][0]                  
    ==================================================================================================
    Total params: 401,700
    Trainable params: 401,700
    Non-trainable params: 0
    __________________________________________________________________________________________________
    


```python
wnc.fit(X_train, y_train, epochs = 10, batch_size = 32, optimizer='adam', save=True, save_dir='results/')
```

    Epoch 1/10
    10/10 [==============================] - 35s 3s/step - loss: 2.4053 - acc: 0.1000
    
    Epoch 00001: loss improved from inf to 2.40526, saving model to results/saved_wavenet_clasifier.h5
    Epoch 2/10
    10/10 [==============================] - 30s 3s/step - loss: 2.3023 - acc: 0.1000
    
    Epoch 00002: loss improved from 2.40526 to 2.30232, saving model to results/saved_wavenet_clasifier.h5
    Epoch 3/10
    10/10 [==============================] - 30s 3s/step - loss: 2.3035 - acc: 0.1000
    
    Epoch 00003: loss did not improve from 2.30232
    Epoch 4/10
    10/10 [==============================] - 30s 3s/step - loss: 2.3025 - acc: 0.1000
    
    Epoch 00004: loss did not improve from 2.30232
    Epoch 5/10
    10/10 [==============================] - 28s 3s/step - loss: 2.3026 - acc: 0.1000
    
    Epoch 00005: loss did not improve from 2.30232
    Epoch 6/10
    10/10 [==============================] - 30s 3s/step - loss: 2.3026 - acc: 0.1000
    
    Epoch 00006: loss did not improve from 2.30232
    Epoch 7/10
    10/10 [==============================] - 27s 3s/step - loss: 2.3026 - acc: 0.1000
    
    Epoch 00007: loss did not improve from 2.30232
    Epoch 8/10
    10/10 [==============================] - 29s 3s/step - loss: 2.3026 - acc: 0.1000
    
    Epoch 00008: loss did not improve from 2.30232
    Epoch 9/10
    10/10 [==============================] - 28s 3s/step - loss: 2.3026 - acc: 0.1000
    
    Epoch 00009: loss did not improve from 2.30232
    Epoch 10/10
    


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-35-7aa47dac12b3> in <module>
    ----> 1 wnc.fit(X_train, y_train, epochs = 10, batch_size = 32, optimizer='adam', save=True, save_dir='results/')
    

    ~\Desktop\wavenet-classifier-master\WaveNetClassifier.py in fit(self, X, Y, validation_data, epochs, batch_size, optimizer, save, save_dir)
        176     self.model.compile(optimizer, loss, metrics)
        177     try:
    --> 178       self.history = self.model.fit(X, Y, shuffle = True, batch_size=batch_size, epochs = epochs, validation_data = validation_data, callbacks=callbacks, initial_epoch=self.start_idx)
        179     except:
        180       if save:
    

    ~\Anaconda3\envs\wavenet-class\lib\site-packages\keras\engine\training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, **kwargs)
       1037                                         initial_epoch=initial_epoch,
       1038                                         steps_per_epoch=steps_per_epoch,
    -> 1039                                         validation_steps=validation_steps)
       1040 
       1041     def evaluate(self, x=None, y=None,
    

    ~\Anaconda3\envs\wavenet-class\lib\site-packages\keras\engine\training_arrays.py in fit_loop(model, f, ins, out_labels, batch_size, epochs, verbose, callbacks, val_f, val_ins, shuffle, callback_metrics, initial_epoch, steps_per_epoch, validation_steps)
        197                     ins_batch[i] = ins_batch[i].toarray()
        198 
    --> 199                 outs = f(ins_batch)
        200                 outs = to_list(outs)
        201                 for l, o in zip(out_labels, outs):
    

    ~\Anaconda3\envs\wavenet-class\lib\site-packages\keras\backend\tensorflow_backend.py in __call__(self, inputs)
       2713                 return self._legacy_call(inputs)
       2714 
    -> 2715             return self._call(inputs)
       2716         else:
       2717             if py_any(is_tensor(x) for x in inputs):
    

    ~\Anaconda3\envs\wavenet-class\lib\site-packages\keras\backend\tensorflow_backend.py in _call(self, inputs)
       2673             fetched = self._callable_fn(*array_vals, run_metadata=self.run_metadata)
       2674         else:
    -> 2675             fetched = self._callable_fn(*array_vals)
       2676         return fetched[:len(self.outputs)]
       2677 
    

    ~\Anaconda3\envs\wavenet-class\lib\site-packages\tensorflow\python\client\session.py in __call__(self, *args, **kwargs)
       1380           ret = tf_session.TF_SessionRunCallable(
       1381               self._session._session, self._handle, args, status,
    -> 1382               run_metadata_ptr)
       1383         if run_metadata:
       1384           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)
    

    KeyboardInterrupt: 



```python
test_path = 'TRAIN_WAV/FBJL0/SI2182'
```
