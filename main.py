#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 21:40
# @Author  : Bing
# @email    : 1747193328@qq.com

import numpy as np
import utils
import os
import pandas as pd
from features import random_perovskites,  MyFeatures


def predict(feature_dataframe:pd.DataFrame):
    bandgap_model = utils.load_model("./model/bandgap.json")
    aca_model = utils.load_model("./model/aca.json")
    feature_dataframe.loc[:, "band_gap"] = np.nan
    feature_dataframe.loc[:, "aca"] = np.nan
    bgp_data = feature_dataframe.loc[:, ["$IE_A$", "$LUMO_B$", "$V^d_B$", "$X_B$", "$IE_B$", "$EA_B$", "$Mn_B$", "$V^s_B$", "$X_X$", "$M_X$"]]

    band_gap = bandgap_model.predict(bgp_data.values)
    result = np.where(band_gap < 0, 0, band_gap)
    feature_dataframe["band_gap"] = result

    aca_data = feature_dataframe.loc[:, ["$IE_A$", "$X_B$", "$EA_B$", "$IE_B$", "$Ep_B$", "$V^d_B$", "$V^p_B$", "$HLGap_B$", "$X_X$"]]

    aca = aca_model.predict(aca_data.values)
    result = np.where(aca < 0, 0, aca)
    feature_dataframe["aca"] = result

    return feature_dataframe




if __name__ == '__main__':


    random_list = []
    a_list = ['Rb', 'Cs', ]
    b1_list = ['Li', 'Na', 'K', 'Rb', 'Ag', "Cu"]

    b2_list = ['Mg', 'Ca', 'Ti', 'Cr',
               'Mn', 'Cu',
               'Zn', 'Sr',
               'Ge',
               'Sn', 'Ba', 'Sm', 'Tm', 'Yb',
               ]
    b3_list = ['Al', 'Sc', 'Bi', "In",

               ]
    x_list = ["F", "Cl", "Br", "I"]
    print("A random perovskite structure is being generated. If there are too many specified elements, please wait patiently.")

    for i in random_perovskites(
            a_list=a_list,
            b1_list=b1_list,
            b2_list=b3_list,
            x_list=x_list,
            doping_ratio=[[1], [0.5], [0.25, 0.125, 0.125], [3]], _filter=True):
        random_list.append({"system": i, "type": "mix-b3"})


    # for i in random_perovskites(
    #         a_list=a_list,
    #         b1_list=b1_list,
    #         b2_list=b3_list,
    #         doping_ratio=[[1], [0.25, 0.125, 0.125], [0.5], [3]], _filter=True):
    #     random_list.append({"system": i, "type": "mix-b1"})
    # for i in random_perovskites(
    #         a_list=a_list,
    #         b1_list=b2_list,
    #         b2_list=b2_list,
    #         doping_ratio=[[1], [0.5], [0.25, 0.125, 0.125], [3]], _filter=True):
        random_list.append({"system": i, "type": "mix-b2"})
    feature_calculators = MyFeatures()
    feature_calculators.set_n_jobs(os.cpu_count())
    datas = pd.DataFrame(random_list)
    datas = datas.drop_duplicates(["system"])
    print("Start generating feature descriptors")
    feature_data = feature_calculators.featurize_dataframe(datas, col_id='system', ignore_errors=True)

    result=predict(feature_data)
    print(f"A total of {result.shape[0]} perovskite structures were generated")
    result.loc[:,["system","band_gap","aca"]].sort_values("aca",ascending=False).to_csv("./all_predict_result.csv")

    filter_data = result[(result.band_gap > 0) & (result.band_gap < 1.5)&(result.aca >=4.5)]


    filter_data.loc[:,["system","band_gap","aca"]].sort_values("aca",ascending=False).to_csv("./filter_predict_result.csv")
    print("The number of eligible perovskites is ",filter_data.shape[0])