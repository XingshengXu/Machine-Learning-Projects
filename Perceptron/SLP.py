import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_set = idx2numpy.convert_from_file('./dataset/train-images.idx3-ubyte')
label_set = idx2numpy.convert_from_file('./dataset/train-labels.idx1-ubyte')
