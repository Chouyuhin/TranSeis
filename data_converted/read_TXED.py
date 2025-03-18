import h5py
import numpy as np

# 定义输入和输出文件名
input_filename = "D:/Google/Downloads/TXED_20231111.h5"
output_filename = "D:/Google/Downloads/TXED_20231111_modified.h5"
ids=np.load("D:/Internet Explorer/downloads/ID_20231111.npy")
# 指定要处理的数据集名称
ev_names = [name for name in ids if name.endswith("EV")] 

# 打开输入文件
with h5py.File(input_filename, "r") as input_file:
    #print(input_file.keys())
    # 创建一个新的输出文件
    with h5py.File(output_filename, "w") as output_file:
        # 遍历指定的前五个数据集
        for dataset_name in ev_names:
            if dataset_name in input_file:
                # 获取当前数据集
                dataset = input_file[dataset_name]
                data = np.array(dataset['data'])  # 读取原始数据
                #print(data.shape)

                # 检查原始数据是否符合预期的形状 (6000, 3)
                if data.shape == (6000, 3):
                    # 创建一个新的 (20000, 3) 的矩阵，前后填充 7000 行 0
                    padded_data = np.zeros((20000, 3), dtype=data.dtype)
                    padded_data[:6000, :] = data  # 将原数据放入中间

                    # 在输出文件中创建相应的数据集，并写入填充后的数据
                    grp = output_file.create_group(dataset_name)
                    grp.create_dataset('data', data=padded_data)

                    # 如果原数据集中还有其他属性，复制它们
                    for attr_name, attr_value in dataset.attrs.items():
                        grp.attrs[attr_name] = attr_value
                else:
                    print(f"Warning: Dataset {dataset_name} has unexpected shape {data.shape}, skipping.")
            else:
                print(f"Dataset {dataset_name} not found in the input file, skipping.")