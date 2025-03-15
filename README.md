# TranSeis

## requirments

```
# requirements
python == 3.7.1
tensorflow == 2.5.3
keras == 2.3.1
h5py == 3.1.0
numpy == 1.19.1
tqdm == 4.66.2
scipy == 1.7.3 
pandas == 1.3.5 
obspy == 1.3.1 

```

<br/>

## Code explanation
- data_converted：data preproessing
    - convert_mseed_to_h5_basic.py：converting mseed format to h5 format without changing the waveform data itself
    - convert_mseed2hdf5_tranc.py：read single or multiple miniseed files + view group names of hdf5 files + slice mseed waveforms and write to hdf5 in desired format + blend into one file by station name or month
    - generate_testnpy.ipynb：genarate test.npy
    - hdf5_reader.py：read and plot waveforms in h5 file

- generator：
    - cal_epicenter.ipynb：simple calculation of predicted epicenter distance using s-p
    - Method.py：model structure + other modelus
    - predictor_final.py
    - tester.py
    - trainer.py

- prompt
    - inference_final
    - testing_py.py
    - training_py.py





<br/>

## Notes
Some notes that need to be modified in practice
1. DataGeneratorTest的dataset= fl.get(str(ID))，# group name of your own h5 file

    DiTing.h5 group_name：['earthquake']

    DiTing2.0.h5 group_name：['data']

    Qiaojia.h5 group_name：['earthquake']


## Paper
Zhou, Y., Zhang, H., Chen, S., Yuan, Z., Tan, C., Huang, F., Guo, Y., Shi, Y., 2025. TranSeis: A high precision multitask seismic waveform detector. Computers & Geosciences 196, 105867. https://doi.org/10.1016/j.cageo.2025.105867
Link：https://doi.org/10.1016/j.cageo.2025.105867
