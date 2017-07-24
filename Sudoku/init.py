#!/usr/bin/env python
# encoding: utf-8
"""
init.py

Created by Safak Ozkan on 2017-05-04.
Copyright (c) 2017 __MyCompanyName__. All rights reserved.
"""


assignments = []
boxes = []
rows = 'ABCDEFGHI'
cols = '123456789'
def cross(A, B):
    return [s+t for s in A for t in B]

boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
diag_units = [[rows[i]+cols[i]  for i in range(9)]] + [[rows[i]+cols[8-i]  for i in range(9)]] 

unitlist = row_units + column_units + square_units + diag_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)
