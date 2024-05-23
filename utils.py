#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/22 21:40
# @Author  : Bing
# @email    : 1747193328@qq.com
import logging
from xgboost import XGBRegressor


def load_model(model_path) ->XGBRegressor:

    model = XGBRegressor()
    model.load_model(model_path)
    return model


def catch_ex(func):
    def wrap(*args):
        try:
            result = func(*args)
        except KeyError as e:
            result = None
            logging.warning(f"{func}\n在数据文件里，并没有{args[1]}数据，请补充一下：{e.args[0]}")
            # raise KeyError()
        return result

    return wrap


