# -*- coding: utf-8 -*-
"""
Created on Thu May 28 02:34:59 2020

@author: User
"""
import addmet

data = None
model_type = None
params_list = None

optres = addmet.AICOptimizer(model_type, 1, 1, 1, data, params_list=params_list, exact=False)