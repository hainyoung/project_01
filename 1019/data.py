# import pandas as pd
# import numpy as np

# # df = pd.read_csv('./output_1012.csv', encoding = 'utf-8')
# # df = pd.read_csv('./1015_output_no_frame_shifts_5frames.csv', encoding = 'utf-8')

# print(df)
# print("==========================================")
# print(df.info())
# # <class 'pandas.core.frame.DataFrame'>
# # RangeIndex: 8522 entries, 0 to 8521
# # Data columns (total 5 columns):
# #  #   Column  Non-Null Count  Dtype
# # ---  ------  --------------  -----
# #  0   date    8522 non-null   object
# #  1   time    4297 non-null   object
# #  2   speed   4642 non-null   float64
# #  3   lane    8522 non-null   int64
# #  4   auto    8522 non-null   int64
# # dtypes: float64(1), int64(2), object(2)
# # memory usage: 333.0+ KB
# # None
# print("==========================================")
# print(df.describe())
# #              speed         lane         auto
# # count  4642.000000  8522.000000  8522.000000
# # mean     22.946575     0.085074     0.029688
# # std      29.007658     0.279008     0.169735
# # min       0.000000     0.000000     0.000000
# # 25%       6.000000     0.000000     0.000000
# # 50%       8.000000     0.000000     0.000000
# # 75%      36.000000     0.000000     0.000000
# # max     971.000000     1.000000     1.000000
# print("==========================================")
# print(df.count())
# # date     8522
# # time     4297
# # speed    4642
# # lane     8522
# # auto     8522
# # dtype: int64
# print("==========================================")
# print(df.notnull())
# print("==========================================")
# print(df.isnull())
# print("==========================================")

# print(df.isnull().sum())
# # date        0
# # time     4225
# # speed    3880
# # lane        0
# # auto        0


