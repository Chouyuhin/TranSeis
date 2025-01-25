import obspy
import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import ipdb

'''
# 读取单个MiniSEED文件
mseed_file = 'g:/云南巧家宽频带数据/01/2022/08/QJ.QJ01._centaur-3_8480_20220825_040000.miniseed'
st = obspy.read(mseed_file)

# 打印读取的数据信息
print(st)

# 打印trace数量
print(len(st))

# 可视化波形数据
st.plot()
#st.plot( starttime=st[0].stats.starttime, endtime= st[0].stats.starttime + 199)
'''



'''
# 读取多个MiniSEED文件
mseed_files = ['g:/云南巧家宽频带数据/01/2023/02/QJ.QJ01._centaur-3_8480_20230201_000000.miniseed','g:/云南巧家宽频带数据/01/2023/02/QJ.QJ01._centaur-3_8480_20230201_010000.miniseed','g:/云南巧家宽频带数据/01/2023/02/QJ.QJ01._centaur-3_8480_20230201_020000.miniseed']
# 读取所有MiniSEED文件
streams = []
for mseed_file in mseed_files:
    st = obspy.read(mseed_file)
    streams.append(st)

# 合并所有Stream对象
combined_stream = sum(streams, obspy.Stream())

# 打印读取的数据信息
print(combined_stream)

# 可视化波形数据
# combined_stream.plot()   # 可视化所有波形

# 自定义可视化选项,60以s为单位
combined_stream.plot( starttime=combined_stream[0].stats.starttime + 4800, endtime=combined_stream[0].stats.starttime + 5000)

'''


'''
## 查看示例hdf5文件的组名字  earthquake/QJ
def list_hdf5_groups(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        group_list = []
        def print_name(name):
            group_list.append(name)
            # print(name)
        # 使用 f.keys() 获取顶层组和数据集名称
        #return list(f.keys())      # ['earthquake']
        
        # 使用visit方法遍历文件中的所有对象
        f.visit(print_name)
    return group_list
    

        

# 示例调用
hdf5_file = 'g:/云南巧家宽频带数据/01/2023/02/hdf5/202302.hdf5'
list_hdf5_groups(hdf5_file)
'''



### main code for oncvertinf mseed to hdf5 ###
# 定义函数用于切分数据段
def split_segments(stream, segment_length):
    segments = []
    
    total_duration = stream[0].stats.endtime - stream[0].stats.starttime   # len(stream)=3
    start_time = stream[0].stats.starttime
    #print(stream[0].stats.endtime)
    
    
    # 切分数据段
    while start_time + segment_length <= stream[0].stats.endtime+0.01:
        #print(start_time + segment_length)
        end_time = start_time + segment_length-0.01
        segment = stream.slice(start_time, end_time)     # len(segment)=3
        segments.append(segment)     # segments=[(20000,3),...(20000,3)]
        start_time += segment_length
    
    #group_names = [f"{group_name_prefix}{i + 1:02}" for i in range(len(segments))]
    
    return segments

folder_path ='g:/云南巧家宽频带数据'    # 文件夹路径
# 获取文件夹中所有 MiniSEED 文件的路径
#
# = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.miniseed')]

#hdf5_file = h5py.File('g:/云南巧家宽频带数据/hdf5/02.hdf5','w')

segment_length = 200  # 每段长度为200秒
output_dir = 'g:/云南巧家宽频带数据/hdf5_merged'



# 遍历文件夹中的所有文件
#ipdb.set_trace()
# for root, station_dirs, files in os.walk(folder_path):
#     for num in station_dirs[:10]:
#         #ipdb.set_trace()
#         station_path = os.path.join(folder_path, num)   # 'g:/云南巧家宽频带数据\\01'
#         hdf5_file = h5py.File(os.path.join(output_dir, f"QJ{num}.hdf5"),'w')    # 'g:/云南巧家宽频带数据/hdf5_merged\\QJ01.hdf5'
#         for _, year_dirs, _ in os.walk(station_path):
#             for year in year_dirs:
#                 year_path = os.path.join(station_path, year)     # 'g:/云南巧家宽频带数据\\01\\2022'
#                 for _, month_dirs, _ in os.walk(year_path):
#                     for month in month_dirs:
#                         month_path = os.path.join(year_path, month)     # 'g:/云南巧家宽频带数据\\01\\2022\\08'
#                         if not os.path.exists(month_path):
#                             print(f"路径不存在：{month_path}")
#                         # ipdb.set_trace()
#                         else:
#                             mseed_files = [os.path.join(month_path, f) for f in os.listdir(month_path) if f.endswith('.miniseed')]
                
                
# hdf5_files = {}
mseed_files_dict = {f'QJ{i:02}': [] for i in range(1, 11)}    # {'QJ01': [], 'QJ02': [], 'QJ03': [], 'QJ04': [], 'QJ05': [], 'QJ06': [], 'QJ07': [], 'QJ08': [], 'QJ09': [], 'QJ10': []}
#ipdb.set_trace()
station_pattern = r'QJ0[1-9]|QJ10' 

#ipdb.set_trace()
# 收集所有 .mseed 文件的路径
for station in os.listdir(folder_path):
    station_path = os.path.join(folder_path,station)
    # mseed_files_dict = {station:[]}
    
    for root, dirs, files in os.walk(station_path):
        root_name = os.path.basename(root)
        for file_name in files:
            if file_name.endswith('.miniseed'):
                mseed_files_dict[station].append( os.path.join(root, file_name))  
                # {'QJ01': ['g:/云南巧家宽频带数据\\QJ01\\2022\\08\\QJ.QJ01._centaur-3_8480_20220825_040000.miniseed', 'g:/云南巧家宽频带数据\\QJ01\\2022\\08\\QJ.QJ01._centaur-3_8480_20220825_050000.miniseed'], 
                # 'QJ02': [], 'QJ03': [], 'QJ04': [], 'QJ05': [], 'QJ06': [], 'QJ07': [], 'QJ08': [], 'QJ09': [], 'QJ10': []}
            
    # if re.match(station_pattern, root_name):  # 检查根目录名是否为01到10
    #     if root_name not in hdf5_files:
    #         # 创建相应的 HDF5 文件
    #         hdf5_files[root_name] = h5py.File(f'segments_{root_name}.h5', 'w')    # {'QJ01': <HDF5 file "segments_QJ01.h5" (mode r+)>}
            

            # 逐个处理每个 MiniSEED 文件
    hdf5_file = h5py.File(f'g:/云南巧家宽频带数据/hdf5_merged/{station}.hdf5','w')
    for file_index, file_name in enumerate(mseed_files_dict[station]):

        # 提取日期时间部分，假设文件名格式为 QJ.QJ01._centaur-3_8480_20220825_040000.miniseed
        base_name = os.path.basename(file_name)
        parts = base_name.split('_')
        station_id = parts[0]
        date_time_part = parts[-2]  # 提取倒数第二部分，即日期时间部分
        hour_part = parts[-1].split('.')[0]
        
        # 读取 MiniSEED 文件
        stream = obspy.read(file_name)
        #ipdb.set_trace()
        # 切分数据段
        segments = split_segments(stream, segment_length)

        # 关闭所有 HDF5 文件
   
        
        # 将每个数据段存储到 HDF5 文件中
        
        for segment_index, segment in enumerate(segments):
            # 组名示例：earthquake/QJ01000001.0001    segment格式仍然是(samples，channels)
            group_name = f"earthquake/{station_id}{date_time_part}.{hour_part[0:4]}{segment_index + 1:02}"
            #print(group_name)
            # 检查数据集是否已经存在
            if group_name in hdf5_file:
                print(f"Dataset {group_name} already exists. Skipping...")
                continue
            data = np.zeros((segment_length*100, len(segment)), dtype=np.float32)
            
            # 将波形数据转换为 numpy 数组
            for trace_index, trace in enumerate(segment):
                if len(trace.data) != segment_length*100:
                    #ipdb.set_trace()
                    trace_length = min(segment_length*100, len(trace.data))
                    data[:trace_length, trace_index] = trace.data[:trace_length]
                    
                    
                else:    
                    data[:, trace_index] = trace.data
                    
            
            
            # 创建 dataset 并写入数据
            hdf5_file.create_dataset(group_name, data=data)
        
        print(f"Processed {file_name}. Total segments: {len(segments)}")
    # 关闭 HDF5 文件
    hdf5_file.close()
           


print("All segments saved to HDF5 file.")





'''
"""
    merge files
    将每个台站不同月的融合成一个h5文件,按台站命名
"""
def merge_hdf5_files(input_files, output_file):
    with h5py.File(output_file, 'w') as f_out:
        for file in input_files:
            with h5py.File(file, 'r') as f_in:
                def copy_obj(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        #print(f"以dataset存储的:{name}")
                        if name in f_out:
                            # 如果数据集已存在，跳过或处理冲突
                            print(f"Dataset {name} already exists. Skipping.")
                        else:    
                            # 复制数据集到新文件中
                            f_out.create_dataset(name, data=obj[:])
                    elif isinstance(obj, h5py.Group):
                        print(f"以组存储的：{name}")
                        # 创建新的组，并递归复制其中的对象
                        if name not in f_out:
                            f_out.create_group(name)
                        for sub_name, sub_obj in obj.items():
                            copy_obj(f"{name}/{sub_name}", sub_obj)

                f_in.visititems(copy_obj)

                


input_dir = "G:/云南巧家宽频带数据/hdf5"
input_files = os.listdir(input_dir)




# 按文件前缀分组
files_grouped = {}
for file in input_files:
    if file.endswith('.hdf5'):
        prefix = file.split('_')[0]
        if prefix not in files_grouped:
            files_grouped[prefix] = []
        files_grouped[prefix].append(os.path.join(input_dir, file))

# 合并文件
output_dir = "G:/云南巧家宽频带数据/hdf5_merged"  # 替换为你的输出目录路径
for prefix, files in files_grouped.items():
    output_file = os.path.join(output_dir, f"QJ{prefix}.hdf5")
    merge_hdf5_files(files, output_file)
    # list_hdf5_groups(output_file)
    print(f"Merged files {files} into {output_file}")
    
'''

