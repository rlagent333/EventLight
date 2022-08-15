#!/usr/bin/env python
# coding=utf8
# File: del.py
import json
import numpy as np
f = open('./flow.json')
flow = json.load(f)
rest_data = []
for car in flow:
  if np.random.random() < 0.8:
    rest_data.append(car)
json_object = json.dumps(rest_data)
with open('./eval_flow.json', 'w') as out_file:
  out_file.write(json_object)
