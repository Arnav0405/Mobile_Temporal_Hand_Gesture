import pandas as pd
import numpy as np

file = "./dataset/idle_gestures/seq_0.npy"

sequence = np.load(file)
print(sequence.shape)
print(sequence[0])  # Print the first frame of the sequence
print(sequence[0].shape)  # Print the shape of the first frame
