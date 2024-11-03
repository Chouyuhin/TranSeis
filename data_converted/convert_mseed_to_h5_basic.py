import obspy
import h5py
import numpy as np

def convert_mseed_to_hdf5(mseed_file, hdf5_file):
    # 读取MSEED文件
    st = obspy.read(mseed_file)
    
    # 打开一个新的HDF5文件
    with h5py.File(hdf5_file, 'w') as f:
        for i, tr in enumerate(st):
            # 创建一个组来存储每个Trace
            grp = f.create_group(f'trace_{i}')
            
            # 存储数据
            grp.create_dataset('data', data=tr.data)
            
            # 存储元数据
            grp.attrs['station'] = tr.stats.station
            grp.attrs['network'] = tr.stats.network
            grp.attrs['location'] = tr.stats.location
            grp.attrs['channel'] = tr.stats.channel
            grp.attrs['starttime'] = str(tr.stats.starttime)
            grp.attrs['sampling_rate'] = tr.stats.sampling_rate

# 示例调用
mseed_file = 'path/to/your/input.mseed'
hdf5_file = 'path/to/your/output.hdf5'
convert_mseed_to_hdf5(mseed_file, hdf5_file)
