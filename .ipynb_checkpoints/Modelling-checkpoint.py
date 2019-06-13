import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os

datadir = "data/"

os.listdir(datadir)
train = pd.read_csv("data/training.csv")
test = pd.read_csv("data/test.csv")

print(train.head())
print(test.head())
