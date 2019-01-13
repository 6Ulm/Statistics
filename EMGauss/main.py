import EM_K_means, os

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

# read and convert data to numpy array
def to_numpy(file_path, filename):
    with open(file_path + filename, 'r') as file:
        data = file.readlines()
    data = [line.replace('\n', '') for line in data]
    data = [line.split() for line in data]
    data = np.array([list(map(float, line)) for line in data])
    return data

# import data
file_train = "EMGaussian.data"
file_test = "EMGaussian.test"
train = to_numpy(file_path, file_train)
test = to_numpy(file_path, file_test)
data_set = {'train': train,
            'test': test}

# Plot Kmeans
EM_K_means.plot_kmeans_clusters(train, general_para['K'], \
                        general_para['n_init'], save_path)

# Plot EM with General covariance matrix
EM_K_means.plot_EM_clusters(data_set, general_para, False, save_path)

# Plot EM with Isotropic covariance matrix
EM_K_means.plot_EM_clusters(data_set, general_para, True, save_path)
