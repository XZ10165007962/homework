# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         hw3
# Description:
# Author:       xinzhuang
# Date:         2022/3/13
# Function:
# Version：
# Notice: 图像分类的卷积神经网络
# -------------------------------------------------------------------------------
# !/usr/bin/python3
# -*- coding:utf-8 -*-

import math

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import matplotlib.pyplot as plt

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)


def all_date():
    """
    生成完整时间轴数据
    """
    dates = pd.DataFrame(data={"sales_date_new": pd.date_range(start="2019-01-01", end="2022-12-31")})
    dates["week"] = dates.sales_date_new.dt.week  # 月份
    dates["year"] = dates.sales_date_new.dt.year  # 距离2020年的年限
    dates["date_block_num"] = (dates["year"] - 2019) * 52 + dates["week"]  # 售卖的月份
    dates["first_day_of_month"] = dates.sales_date_new.dt.dayofyear
    # dates["days_num"] = dates.groupby(["year", "month"])["first_day_of_month"].transform("count")  # 每个月的天数
    del dates["first_day_of_month"]
    del dates["sales_date_new"]
    return dates, [], []


def read_data(path, columns):
    """
    读取数据
    :param path: 路径
    :param columns: 使用的数据列名
    :return:
    """
    data_ = pd.read_csv(path)
    data = data_.loc[:, columns]
    skus = [
        130100002187, 130100002188, 130100000985, 130100001722, 130100002118, 130100002120, 130100002136, 130100002782,
        130100002074, 130100001090, 130100001735, 130100000612, 130100000613, 130100000614, 130100000615, 130100000177,
        130100000179, 130100000181, 130100000138, 130100000141, 130100001632, 130100002732, 130100002715, 130100000708,
        130100000709, 130100001982, 130100000912, 130100000914]
    data = data[data["material_code"] == 130100000912]
    data.dropna(subset=columns, how="any", inplace=True)
    data.sort_values("order_create_date").reset_index(drop=True, inplace=True)
    # pro_data = read_class_data()
    # pro_data = pro_data.loc[:, ["sap_code", "category_desc"]]
    # data = data.merge(pro_data, left_on="material_code", right_on="sap_code", how="left")

    return data


def read_class_data():
    """
    读取产品主数据
    :return:
    """
    product_data_path = "/Users/xinzhuang/Documents/特通/数据/product_main_data_20220323.csv"
    product_data = pd.read_csv(product_data_path)
    product_data = product_data.loc[:,
                   ["sap_code", "material_desc", "material_group_desc", "product_group_desc", "specification",
                    "fin_mgmt_report_brand_desc", "classify_desc", "category_desc", "brand_desc", "brand_group_desc"]]
    return product_data


def trans_data(df):
    """
    对时间格式数据进行数据转换，并对原始字段进行统计分析
    :param df: 输入数据
    :return:
    """
    continu_feat = []
    cate_feat = []

    df["order_create_date"] = pd.to_datetime(df["order_create_date"].astype("str"))
    df["year_month"] = df["order_create_date"].dt.strftime("%Y%m")
    df["year"] = df["order_create_date"].dt.year
    df["month"] = df["order_create_date"].dt.month
    df["is_quarter_start"] = df["order_create_date"].dt.is_quarter_start
    df["is_quarter_end"] = df["order_create_date"].dt.is_quarter_end
    df["is_year_start"] = df["order_create_date"].dt.is_year_start
    df["is_year_end"] = df["order_create_date"].dt.is_year_end
    df["quarter"] = df["order_create_date"].dt.quarter
    df["week"] = df["order_create_date"].dt.week
    cate_feat += ["month", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end",
                  "quarter", "week"]
    # 添加月份标记字段
    date, feat1, feat2 = all_date()
    cate_feat += feat1
    continu_feat += feat2

    df = df.merge(date, on=["year", "week"], how="left")
    df.sort_values(["order_create_date"], inplace=True)
    "========================================sku数据统计============================================"
    df["qty_sum"] = df.groupby(["year", "week", "material_code"])["order_qty"].transform("sum")
    df["qty_mean"] = df.groupby(["year", "week", "material_code"])["order_qty"].transform("mean")
    df["qty_std"] = df.groupby(["year", "week", "material_code"])["order_qty"].transform("std")

    # # 对数据进行log变化
    # df["qty_sum"] = list(map(lambda x: np.log(x + 1), df["qty_sum"]))
    # df["qty_mean"] = list(map(lambda x: np.log(x + 1), df["qty_mean"]))
    # df["qty_std"] = list(map(lambda x: np.log(x + 1), df["qty_std"]))
    continu_feat += ["qty_sum", "qty_mean", "qty_std"]

    # 计算当月该sku对应的客户数量以及送达发数量
    month_ship_nums = df.groupby(["year", "week"])["ship_toparty"].nunique()
    month_sold_nums = df.groupby(["year", "week"])["sold_toparty"].nunique()
    month_ship_nums_pd = pd.DataFrame(month_ship_nums).reset_index()
    month_sold_nums_pd = pd.DataFrame(month_sold_nums).reset_index()
    month_ship_nums_pd = month_ship_nums_pd.rename(columns={"ship_toparty": "month_ship_nums"})
    month_sold_nums_pd = month_sold_nums_pd.rename(columns={"sold_toparty": "month_sold_nums"})

    df = df.merge(month_ship_nums_pd, on=["year", "week"], how="left")
    df = df.merge(month_sold_nums_pd, on=["year", "week"], how="left")

    continu_feat += ["month_ship_nums", "month_sold_nums"]

    df_mat = df.drop_duplicates(["year", "week", "material_code"])
    df_mat.reset_index(drop=True, inplace=True)

    return df_mat, cate_feat, continu_feat


def get_feature(df):
    """
    构建数据特征
    :param df: 输入数据
    :return:
    """
    continu_feat = []
    cate_feat = []

    # 假期信息
    # jiaqibiao = [
    #     [9, 11, 10, 8, 10, 11, 8, 9, 9, 12, 9, 9], [14, 9, 9, 8, 12, 9, 8, 10, 7, 13, 9, 8],
    #     [11, 10, 8, 8, 12, 9, 9, 9, 8, 14, 8, 8], [10, 12, 8, 9, 11, 9, 10, 8, 9, 13, 8, 9]]
    # df['count_jiaqi'] = list(
    #     map(lambda x, y: jiaqibiao[int(x) - 2019][int(y - 1)], data['year'], data['month']))
    # continu_feat.append("count_jiaqi")

    # 春节、端午、中秋月份标记
    df["is_chunjie"] = list(
        map(lambda x: 1 if x == 2 or x == 12 or x == 26 or x == 38 else 0, df["date_block_num"])
    )
    df["is_chunjie_before"] = list(
        map(lambda x: 1 if x == 1 or x == 11 or x == 25 or x == 37 else 0, df["date_block_num"])
    )
    df["is_duanwu"] = list(
        map(lambda x: 1 if x == 6 or x == 18 or x == 30 or x == 42 else 0, df["date_block_num"])
    )
    df["is_duanwu_before"] = list(
        map(lambda x: 1 if x == 5 or x == 17 or x == 29 or x == 41 else 0, df["date_block_num"])
    )
    df["is_zhongqiu"] = list(
        map(lambda x: 1 if x == 9 or x == 22 or x == 33 or x == 45 else 0, df["date_block_num"])
    )
    df["is_zhongqiu_before"] = list(
        map(lambda x: 1 if x == 8 or x == 21 or x == 32 or x == 44 else 0, df["date_block_num"])
    )

    # 11月份月份标记，数据呈现的结果为每年11月数据都有上升
    # 标记五一，十一月份
    df["super_month"] = list(
        map(lambda x: 1 if x == 5 or x == 10 or x == 11 else 0, df["month"])
    )
    cate_feat += [
        "is_chunjie", "is_chunjie_before", "is_duanwu", "is_duanwu_before", "is_zhongqiu", "is_zhongqiu_before",
        "super_month"
    ]

    "===========================================sku维度相关特征构建==================================================="
    # 构造滞后特征
    lag_n = 6
    columns = ["date_block_num", "qty_sum", "material_code"]
    df["id_lag1"] = df["qty_sum"]
    continu_feat.append("id_lag1")
    for i in range(2, lag_n + 1):
        temp = df[columns]
        temp = temp.groupby(["material_code", "date_block_num"]).agg({"qty_sum": "first"}).reset_index()
        temp.columns = ["material_code", "date_block_num"] + [f"id_lag{i}"]
        temp["date_block_num"] += (i - 1)
        df = pd.merge(df, temp, on=["material_code", "date_block_num"], how="left")
        continu_feat.append(f"id_lag{i}")

    # 半年订单量统计
    # df['1_6_sum'] = df.loc[:, 'id_lag1':'id_lag6'].sum(1)
    # df['1_6_mea'] = df.loc[:, 'id_lag1':'id_lag6'].mean(1)
    # df['1_6_max'] = df.loc[:, 'id_lag1':'id_lag6'].max(1)
    # df['1_6_min'] = df.loc[:, 'id_lag1':'id_lag6'].min(1)
    # df['1_6_std'] = df.loc[:, 'id_lag1':'id_lag6'].std(1)
    # df['jidu_1_3_sum'] = df.loc[:, 'id_lag1':'id_lag3'].sum(1)
    # df['jidu_4_6_sum'] = df.loc[:, 'id_lag4':'id_lag6'].sum(1)
    # df['jidu_1_3_mean'] = df.loc[:, 'id_lag1':'id_lag3'].mean(1)
    # df['jidu_4_6_mean'] = df.loc[:, 'id_lag4':'id_lag6'].mean(1)
    # df['jidu_1_3_std'] = df.loc[:, 'id_lag1':'id_lag3'].std(1)
    # df['jidu_4_6_std'] = df.loc[:, 'id_lag4':'id_lag6'].std(1)
    # continu_feat += [
    #     '1_6_sum', '1_6_mea', '1_6_max', '1_6_min', '1_6_std',
    #     'jidu_1_3_sum', 'jidu_4_6_sum', 'jidu_1_3_mean', 'jidu_4_6_mean',
    #     'jidu_1_3_std', 'jidu_4_6_std'
    # ]
    # 趋势特征
    # df['1_2_diff'] = df['id_lag1'] - df['id_lag2']
    # df['1_3_diff'] = df['id_lag1'] - df['id_lag3']
    # df['2_3_diff'] = df['id_lag2'] - df['id_lag3']
    # df['2_4_diff'] = df['id_lag2'] - df['id_lag4']
    # df['3_4_diff'] = df['id_lag3'] - df['id_lag4']
    # df['3_5_diff'] = df['id_lag3'] - df['id_lag5']
    # df['4_5_diff'] = df['id_lag4'] - df['id_lag5']
    # df['4_6_diff'] = df['id_lag4'] - df['id_lag6']
    # df['5_6_diff'] = df['id_lag5'] - df['id_lag6']
    # df['jidu_1_2_diff'] = df['jidu_1_3_sum'] - df['jidu_4_6_sum']
    # continu_feat += [
    #     '1_2_diff', '1_3_diff', '2_3_diff', '2_4_diff', '3_4_diff',
    #     '3_5_diff', 'jidu_1_2_diff', '4_5_diff', '4_6_diff', '5_6_diff'
    # ]
    #
    # '环比'
    # df['huanbi_1_2'] = df['id_lag1'] / df['id_lag2']
    # df['huanbi_2_3'] = df['id_lag2'] / df['id_lag3']
    # df['huanbi_3_4'] = df['id_lag3'] / df['id_lag4']
    # df['huanbi_4_5'] = df['id_lag4'] / df['id_lag5']
    # df['huanbi_5_6'] = df['id_lag5'] / df['id_lag6']
    # continu_feat += [
    #     'huanbi_1_2', 'huanbi_2_3', 'huanbi_3_4', 'huanbi_4_5', 'huanbi_5_6'
    # ]
    #
    # 'add环比比'
    # df['huanbi_1_2_2_3'] = df['huanbi_1_2'] / df['huanbi_2_3']
    # df['huanbi_2_3_3_4'] = df['huanbi_2_3'] / df['huanbi_3_4']
    # df['huanbi_3_4_4_5'] = df['huanbi_3_4'] / df['huanbi_4_5']
    # df['huanbi_4_5_5_6'] = df['huanbi_4_5'] / df['huanbi_5_6']
    # continu_feat += [
    #     'huanbi_1_2_2_3', 'huanbi_2_3_3_4', 'huanbi_3_4_4_5', 'huanbi_4_5_5_6'
    # ]

    # 添加目标字段
    columns = ["date_block_num", "qty_sum", "material_code"]
    temp = df[columns]
    temp = temp.groupby(["material_code", "date_block_num"]).agg({"qty_sum": "first"}).reset_index()
    temp.columns = ["material_code", "date_block_num"] + ["target"]
    temp["date_block_num"] -= 1
    df = pd.merge(df, temp, on=["material_code", "date_block_num"], how="left")
    df[f"target"] = df[f"target"].fillna(0).astype("float32")
    target_feat = ["target"]

    df = df.dropna(subset=df.columns, axis=0, how="any")

    return df, cate_feat, continu_feat, target_feat


def show_picture(df_, w, b):
    plt.figure()  # 打开一个空画布
    # 样本点
    plt.scatter(df_.index, df_["result"], alpha=0.5)  # scatter:散点图,alpha:"透明度"
    plt.scatter(df_.index, df_["target"], alpha=0.5, c='red')  # scatter:散点图,alpha:"透明度"
    # 直线
    # plt.plot(df_["target"], w * df_["target"] + b, c='red')
    plt.show()


def create_dataset(dataset, look_back=-8):
    train_x = dataset[:look_back, :-1]
    train_y = dataset[:look_back, -1:]
    test_x = dataset[look_back:, :-1]
    test_y = dataset[look_back:, -1:]
    return train_x, train_y, test_x, test_y


class lstm(nn.Module):
    def __init__(self, input_size=125, hidden_size=32, output_size=1, num_layer=3):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, X):
        x, _ = self.layer1(X)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)

        return x


if __name__ == '__main__':
    data_path = "/Users/xinzhuang/Documents/特通/数据/detail_data_20220329.csv"

    use_columns = [
        "region", "factory_desc", "sold_toparty", "sold_toparty_name", "ship_toparty", "ship_toparty_name",
        "material_code", "material_describe", "order_create_date", "order_qty"
    ]
    continu_columns = []
    cate_columns = []
    data = read_data(data_path, use_columns)
    data, cate_column1, continu_column1 = trans_data(data)
    data, cate_column2, continu_column2, target_column = get_feature(data)
    continu_columns = continu_column1 + continu_column2
    cate_columns = cate_column1 + cate_column2
    cate_columns_new = []
    all_data = np.array([])

    # 将离散数据进行onehot编码
    for cate in ['month', 'quarter', 'week',]:
        temp = pd.get_dummies(data[cate])
        if all_data.size == 0:
            all_data = temp.values
        else:
            all_data = np.hstack([all_data, temp.values])
    # 将连续数据进行归一化
    StandardScalers = []
    for _ in ['id_lag1', 'id_lag2', 'id_lag3', 'id_lag4', 'id_lag5', 'id_lag6']+target_column:
        StandardScalers.append(StandardScaler())

    for i, continu_column in enumerate(['id_lag1', 'id_lag2', 'id_lag3', 'id_lag4', 'id_lag5', 'id_lag6']+target_column):
        temp = StandardScalers[i].fit_transform(data[continu_column].values.reshape(-1, 1))
        if all_data.size == 0:
            all_data = temp
        else:
            all_data = np.hstack([all_data, temp])
    all_data.astype('float32')
    train_x, train_y, test_x, test_y = create_dataset(all_data, look_back=-4)
    # 将数据转化为lstm能识别的形状
    print(train_x)
    feature_n = all_data.shape[1] - 1
    train_x = train_x.reshape(-1, 1, feature_n)
    print(train_x)
    train_y = train_y.reshape(-1, 1, 1)
    test_x = test_x.reshape(-1, 1, feature_n)
    test_y = test_y.reshape(-1, 1, 1)

    train_x = torch.from_numpy(train_x).to(torch.float32)
    train_y = torch.from_numpy(train_y).to(torch.float32)
    test_x = torch.from_numpy(test_x).to(torch.float32)
    test_y = torch.from_numpy(test_y).to(torch.float32)

    model = lstm(feature_n, hidden_size=16)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # 开始训练
    for e in range(20):
        var_x = Variable(train_x)
        var_y = Variable(train_y)
        # 前向传播
        out = model(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 10 == 0:  # 每 100 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
    x = all_data[:, :-1]
    y = all_data[:, -1:]
    x = x.reshape(-1, 1, feature_n)
    y = y.reshape(-1, 1, 1)

    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)

    model = model.eval()
    var_x = Variable(x)
    var_y = Variable(y)
    pred_test = model(var_x)
    pred_test = pred_test.view(-1).data.numpy()

    # 画出实际结果和预测的结果
    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(all_data[:, -1:], 'b', label='real')
    plt.legend(loc='best')
    plt.show()
    # 反转结果
    # result["target_1"] = list(map(lambda x: math.exp(x) - 1, result["target"]))
    # result["result_1"] = list(map(lambda x: math.exp(x)-1, result["result"]))
    # result.to_csv("result_20220331.csv", index=False, header=True, encoding="utf-8_sig")
