import DM2, os

file_path = os.getcwd() + '/data/'
save_path = os.getcwd()

# General parameters
# K: number of groups
# epsilon: relative error, used as criteria of convergence for EM algo
# n_iter_max: max iterations for EM algo
# n_init: number of initialisations
# bootstrap_size: size of sub bootstrap sample used to initialise parameters in EM
general_para = {'K':4, 
                'epsilon': 0.5/100, 
                'n_iter_max': 100,
                'n_init': 1000, 
                'bootstrap_size': 10}

# import data
file_train = "EMGaussian.data"
file_test = "EMGaussian.test"
train = DM2.to_numpy(file_path, file_train)
test = DM2.to_numpy(file_path, file_test)
data_set = {'train': train,
            'test': test}

# Plot Kmeans
DM2.plot_kmeans_clusters(train, general_para['K'], \
                        general_para['n_init'], save_path)

# Plot EM with General covariance matrix
DM2.plot_EM_clusters(data_set, general_para, False, save_path)

# Plot EM with Isotropic covariance matrix
DM2.plot_EM_clusters(data_set, general_para, True, save_path)