import os
import config_DGMR
from batchcreator_DGMR import get_list_IDs
from datetime import datetime
import numpy as np
import tensorflow as tf

print(f"Create splitting sets with corresponding IDs")
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

start_dt = datetime(2025, 4, 18, 8, 35)  # half hour after first data
end_dt = datetime(2025, 4, 28, 0, 0)
print(f"Start date: {start_dt}")
print(f"End date: {end_dt}")

x_length = 6
y_length = 1
filter_no_rain = 'avg0.01mm'
filename_npy = 'list_IDs200621_avg001mm_train.npy'
filename = config_DGMR.path_code + f"data/{filename_npy}"

print(f"Retrieve IDs")

list_IDs = get_list_IDs(start_dt, end_dt, x_length, y_length, filter_no_rain=filter_no_rain)
print(f"Number of IDs: {len(list_IDs)}")
#print("Result of IDs:")
#print(list_IDs)

flat_ids = [sample[1][0] for sample in list_IDs]
np.save(filename, flat_ids)
print(f"Saved at location: {filename}")
#"""
