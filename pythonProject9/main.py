import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# check time series data
# air_pollution = pd.read_csv('../dataset/air_pollution.csv', parse_dates=['date'])
air_pollution = pd.read_csv('C:/Users/SMXzh/Desktop/try/Y2005M1-2022M33.csv', parse_dates=['Date'], encoding='utf-8')
# air_pollution = air_pollution.dropna()
air_pollution.set_index('Date', inplace=True)
# print(air_pollution.head())

values = air_pollution.values

# groups = [0, 1, 2, 3, 4, 5, 6, 7]
groups = [0]


# i = 1
# # plot each column
# for group in groups:
#     plt.subplot(len(groups), 1, i)
#     plt.plot(values[:, group])
#     plt.title(air_pollution.columns[group], y=0.5, loc='right')
#     i += 1
#
#
# plt.show()
# decomposing

# Decomposing our time series
# use stl method
# Additive Model
# `y(t) =  Trend + Seasonality + Noise`
# series = air_pollution.shanxi[:100]
# result = seasonal_decompose(series, model='multiplicative', period=30)
# result.plot()
# plt.show()
#
def filling_future_points(list, num=10):
    nozero_list = [one for one in list if one != 0]
    before_avg, last_avg = sum(nozero_list[:num]) / num, sum(nozero_list[-1 * num:]) / num
    res_list = []
    for i in range(len(list)):
        if list[i] != 0:
            res_list.append(list[i])
        else:
            tmp = int(num / 2) + 1
            if i <= tmp:
                res_list.append(int(before_avg))
            elif i >= len(list) - tmp:
                res_list.append(int(last_avg))
                slice_list = list[i - tmp:i + tmp + 1]
                res_list.append(int(sum(slice_list) / (num - 1)))
    return res_list


window_size = 5
max_points = 180
filling_points = 7
s = air_pollution['Original'][:max_points + filling_points].rolling(window=window_size).mean()
filling_list = filling_future_points(air_pollution['Original'][max_points - filling_points:max_points],
                                     num=filling_points)
true_result = air_pollution['Original'][max_points:max_points + filling_points]
# s.plot()
for i in range(window_size):
    s[i] = air_pollution['Original'][i]
for i in range(max_points, max_points + filling_points):
    s[i] = filling_list[i - max_points]
#
result1 = seasonal_decompose(s, model='multiplicative', period=20)
#
# # predict future ten points
result1.plot()
plt.show()
#
# def calculate_mse(a, b, lower, upper):
#     error = 0
#     for i in range(lower, upper):
#         error += (a[i] - b[i]) * (a[i] - b[i])
#     return error / (upper - lower)
#
#
# prdict_value = result1.seasonal[-filling_points:] + result1.trend[-filling_points:]
# for i in range(max_points, max_points+filling_points):
#     if pd.isna(prdict_value[i - max_points]):
#         prdict_value[i - max_points] = s[i]
#     else:
#         continue
#
# mse_error = calculate_mse(prdict_value, true_result, 0, len(true_result))
# print(mse_error)


# def filling_future_points(list, num=10):
#     nozero_list = [one for one in list if one != 0]
#     before_avg, last_avg = sum(nozero_list[:num]) / num, sum(nozero_list[-1 * num:]) / num
#     res_list = []
#     for i in range(len(list)):
#         if list[i] != 0:
#             res_list.append(list[i])
#         else:
#             tmp = int(num / 2) + 1
#             if i <= tmp:
#                 res_list.append(int(before_avg))
#             elif i >= len(list) - tmp:
#                 res_list.append(int(last_avg))
#                 slice_list = list[i - tmp:i + tmp + 1]
#                 res_list.append(int(sum(slice_list) / (num - 1)))
#     return res_list


# result1.plot()
# plt.show()
# preprocessing
# predict