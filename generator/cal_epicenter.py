# %%
import pandas as pd

# 读取 CSV 文件
file_path = 'F:/copy/eqt_on_DiTing_DLC-master/eqt_on_DiTing_DLC-master-f3be5eb99ea42ce1cb1c6c53d728dbb47081db73/EQT_on_DiTing/test_tester_outputs_good/true&pred.csv'  # 请将 'your_file.csv' 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 计算 (S_pred - P_pred) * 8
epicenter_pred = (((df['S_pred'] - df['P_pred'])/100) * 8).tolist()

# 输出每一行的计算结果
for value in epicenter_pred:
    print(value)

# %%
import pandas as pd

# 读取 CSV 文件
file_path = 'D:/Anaconda/envs/eqt_on_DiTing/eqt_on_diting/DiTing_datas/DiTing330km_part_0.csv'  # 请将 'your_file.csv' 替换为你的 CSV 文件路径
df = pd.read_csv(file_path)

# 筛选出 P_pick == 4.15 且 S_pick == 7.11 的行
filtered_rows =df[(df['p_pick'] == 4.15) & (df['s_pick'] == 7.11)]

# 打印筛选后的行
#for value in filtered_rows:
print(filtered_rows['dis'])



