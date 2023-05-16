from lightfm.data import Dataset
import dask.dataframe as dd
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
import numpy as np

# Reading files
train_data = dd.read_parquet('~/als_train_set.parquet', engine='pyarrow')
val_data = dd.read_parquet('~/als_val_set.parquet', engine='pyarrow')
test_data = dd.read_parquet('~/als_test_set.parquet', engine='pyarrow')

# Fitting dataset with train user and rms ids
dataset = Dataset()
dataset.fit(train_data['user_id'], train_data['rmsid_int'])

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

# Preparing sparse matrix to train lightfm model
(interactions, weights) = dataset.build_interactions(
    [(x[1]['user_id'], x[1]['rmsid_int'], x[1]['ratings']) for x in train_data.iterrows()])
print(repr(interactions))

# Hyperparameter tuning
params_loss = ['warp', 'bpr']
param_rank = [10, 20, 25]
param_item_alpha = [0.00001, 0.0001, 0.001, 0.01]
param_user_alpha = [0.00001, 0.0001, 0.001, 0.01]

# Initializing model with the parameters in each setting
model = LightFM(loss=params_loss[1], no_components=param_rank[2], item_alpha=param_user_alpha[0],
                user_alpha=param_user_alpha[0])

# Fitting the model with train userid v/s rmsid (sparse) matrix
model.fit(interactions)
print(model)

# Preparing validation interactions
interactions_val, _ = dataset.build_interactions(
    [(x[1]['user_id'], x[1]['rmsid_int'], x[1]['ratings']) for x in val_data.iterrows()])

# Mean Average Precision Calculation
map_val = np.mean(precision_at_k(model, test_interactions=interactions_val, k=100, num_threads=14))
print(map_val)

# Final Model Evaluation
# Preparing test interactions
(interactions_test, _) = dataset.build_interactions([(x[1]['user_id'], x[1]['rmsid_int'], x[1]['ratings']) for x in
                                                     test_data.iterrows()])
map_test = np.mean(precision_at_k(model, test_interactions=interactions_test, k=100, num_threads=14))
print("Final MAP of the model:", map_test)
