#!/usr/bin/bash

python3 triangle.py --dir ./data/000/aligned\ images --dst cos --nam tri-cos
python3 triangle.py --dir ./data/000/aligned\ images --dst euc --nam tri-euc

python3 triangle.py --dir ./data/000/aligned\ images --seg --dst cos --nam tri-seg-cos
python3 triangle.py --dir ./data/000/aligned\ images --seg --dst euc --nam tri-seg-euc

python3 clustering.py --dir ./data/000/aligned\ images --dst cos --nam cos
python3 clustering.py --dir ./data/000/aligned\ images --dst euc --nam euc
python3 clustering.py --dir ./data/000/aligned\ images --dst opt --nam opt

python3 clustering.py --dir ./data/000/aligned\ images --seg --dst cos --nam seg-cos
python3 clustering.py --dir ./data/000/aligned\ images --seg --dst euc --nam seg-euc
python3 clustering.py --dir ./data/000/aligned\ images --seg --dst opt --nam seg-opt