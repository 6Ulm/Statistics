import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal


# read and convert datq to numpy array
def to_numpy(file_path, filename):
    with open(file_path + filename, 'r') as file:
        data = file.readlines()
    data = [line.replace('\n', '') for line in data]
    data = [line.split() for line in data]
    data = np.array([list(map(float, line)) for line in data])
    return data

# K-means algo
def k_means(data, K, n_init):
    n_obs = data.shape[0]
    clusters = np.zeros(K)
    distance = np.zeros((n_obs, K))
    
    list_centroids = []
    list_clusters = []
    array_distortion = np.empty(n_init)
    
    # try n_init different initialisations, then pick the parameters whose distortion is min
    for i in range(n_init):
        new_centroids = data[np.random.choice(n_obs, size=K, replace=False), :]
        old_centroids = new_centroids
        error = 1

        while error != 0:
            # step 1: associate each x to the nearest centroid
            for k in range(K):
                distance[:,k] = np.sum((data - new_centroids[k])**2, axis = 1)
            clusters = np.argmin(distance, axis = 1)

            # step 2: minimize distortion wrt each centroid
            old_centroids = new_centroids
            for k in range(K):
                new_centroids[k] = np.mean(data[clusters == k], axis = 0)
            error = np.sum((old_centroids - new_centroids)**2)
        distortion = sum( np.sum( (data[clusters == k] - new_centroids[k])**2 ) for k in range(K) )
        
        list_centroids.append(new_centroids)
        list_clusters.append(clusters)
        array_distortion[i] = distortion
        
    index_min = np.argmin(array_distortion)
    return {'centroids': list_centroids[index_min], 'clusters': list_clusters[index_min], \
            'distortion': array_distortion[index_min]}

# plot Kmeans clusters
def plot_kmeans_clusters(data, K, n_init, save_path):
    Kmeans = k_means(data, K, n_init)
    fig = plt.figure(figsize=(8,6))
    colors = ["orange", "blue", "green", "pink"]
    for c, k, color in zip(Kmeans['centroids'], range(K), colors):
        group = data[Kmeans['clusters'] == k]
        plt.plot(group[:,0], group[:,1], color = color, marker = 'o', \
                 ms = 5, linestyle = '', label = 'cluster' + str(k+1))
        plt.plot(c[0], c[1], marker = 'o', ms = 8, linestyle = '', color = 'brown')
    plt.title('Kmeans: ' + str(K) + ' clusters with distortion = ' + \
            str(int(Kmeans['distortion'])))
    plt.legend()
    plt.grid()
    fig.savefig(os.path.join(save_path, 'Kmeans'))
    plt.close(fig)

# EM Algo
def update_parameters(data, K, mu, pi, sigma_square, is_isotropic):
    n_obs = data.shape[0]
    new_mu = np.zeros(mu.shape)
    new_pi = np.zeros(pi.shape)
    new_sigma_square = np.zeros(sigma_square.shape)
    
    mixture_density = sum(pi[k] * multivariate_normal(mu[k], sigma_square[k]).pdf(data) for k in range(K))
    for k in range(K):
        alpha_k = pi[k] * multivariate_normal(mu[k], sigma_square[k]).pdf(data)/mixture_density
        new_pi[k] = np.mean(alpha_k)
        new_mu[k] = alpha_k.dot(data)/sum(alpha_k)
        if is_isotropic == False:
            new_sigma_square[k] = (data - new_mu[k]).T.dot(np.multiply(alpha_k.reshape(n_obs,1), \
                                                                       (data - new_mu[k]))) / sum(alpha_k)
        else:
            dimension = data.shape[1]
            sigma_square_k = sum(alpha_k[i] * (data[i] - new_mu[k]).dot(data[i] - new_mu[k]) \
                                 for i in range(n_obs)) / (sum(alpha_k)*dimension)
            new_sigma_square[k] = sigma_square_k * np.identity(dimension)
            
    return (new_mu, new_pi, new_sigma_square)

def mixture_density(pi, mu, sigma_square, data, K):
    # return matrix of likelihood of size NxK (for N observations and K clusters)
    return np.array([pi[k] * multivariate_normal(mu[k], sigma_square[k]).pdf(data) \
                                      for k in range(K)]).T

def EM(data, K, epsilon, n_iter_max, boostrap_size, n_init, is_isotropic):
    n_obs = data.shape[0]
    list_mu = []
    list_sigma_square = []
    list_pi = []
    array_likelihood = np.empty(n_init)
    
    # now try n_init different initialisations, then choose the parameters whose loglikelihood is max.
    for i in range(n_init):
        # initialiser parameters: each time pick a bootstrap sample of size=bootstrap_size,
        # then use its mean/variance as initialisation. No need to randomize pi.
        old_pi = np.ones(K)/K
        old_mu = np.array([np.mean(data[np.random.choice(n_obs, size=boostrap_size, replace=True), :], axis = 0) \
                          for _ in range(K)])
        old_sigma_square = np.array([np.cov(data[np.random.choice(n_obs, size=boostrap_size, replace=True), :].T) \
                                    for _ in range(K)])       
        old_likelihood = np.sum(mixture_density(old_pi, old_mu, old_sigma_square, data, K))
        
        # first update
        new_mu, new_pi, new_sigma_square = update_parameters(data, K, old_mu, old_pi, \
                                                            old_sigma_square, is_isotropic)
        new_likelihood = np.sum(mixture_density(new_pi, new_mu, new_sigma_square, data, K))
        
        # calculate relative error
        relative_error = np.log(new_likelihood)/np.log(old_likelihood)-1
        
        # iterate until relative error < epsilon.
        # n_iter_max is added to avoid too slow convergence
        count = 0
        while abs(relative_error) >= epsilon and count <= n_iter_max:
            old_mu, old_pi, old_sigma_square = new_mu, new_pi, new_sigma_square
            new_mu, new_pi, new_sigma_square = update_parameters(data, K, old_mu, old_pi, \
                                                                 old_sigma_square, is_isotropic)
            old_likelihood = np.sum(mixture_density(old_pi, old_mu, old_sigma_square, data, K))
            new_likelihood = np.sum(mixture_density(new_pi, new_mu, new_sigma_square, data, K))
            relative_error = np.log(new_likelihood)/np.log(old_likelihood)-1
            count += 1
        
        list_mu.append(new_mu)
        list_sigma_square.append(new_sigma_square)
        list_pi.append(new_pi)
        array_likelihood[i] = new_likelihood
    
    index_max = np.argmax(array_likelihood)
    return (list_mu[index_max], list_sigma_square[index_max], \
            list_pi[index_max], np.log(array_likelihood[index_max]))

# plot EM clusters
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots amymn `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].        
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    
    # Plots an `nstd` sigma error ellipse based on 
    # the specified covariance matrix (`cov`)
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def plot_EM_clusters(data_set, general_para, is_isotropic, save_path):
    K = general_para['K']
    epsilon = general_para['epsilon']
    n_iter_max = general_para['n_iter_max']
    bootstrap_size = general_para['bootstrap_size']
    n_init = general_para['n_init']

    mu, sigma_square, pi, loglikelihood = EM(data_set['train'], K, epsilon, \
                            n_iter_max, bootstrap_size, n_init, is_isotropic)

    fig = plt.figure(figsize=(30,12))
    if is_isotropic == True:
        figure_title = 'EM with isotropic covariance matrix'
    else:
        figure_title = 'EM with general covariance matrix'
    fig.suptitle(figure_title, fontsize = 20)

    colors = ["orange", "blue", "brown", "pink"]
    for i, set_name in enumerate(data_set):
        ax = fig.add_subplot(1,2, i+1)

        for i,color in enumerate(colors):
            # generate the ellipsoid corresponding to the parameters
            points = np.random.multivariate_normal(mean=mu[i], cov=sigma_square[i], size=100000)
            x, y = points.T
            ax.plot(x, y, ls = '', marker = 'o', ms = 0.000001, color = 'white')
            plot_point_cov(points, nstd=2, alpha=0.5, color='green')
            
            # get clusters of data
            matrix_likelihood = mixture_density(pi, mu, sigma_square, data_set[set_name], K)
            clusters = np.argmax(matrix_likelihood, axis = 1)
            
            # plot data in each group
            group = data_set[set_name][clusters == i]
            ax.plot(group[:,0], group[:,1], ls = '', marker = 'o', \
                    ms = 5, color = color, label = 'cluster ' + str(i+1))
        
        ax.set_title('Clustering on ' + set_name + ' set with loglikelihood is ' + str(loglikelihood), \
                    fontsize = 18)
        ax.set_xlim((min(data_set[set_name][:,0]) - 2, max(data_set[set_name][:,0]) + 2))
        ax.set_ylim((min(data_set[set_name][:,1]) - 2, max(data_set[set_name][:,1]) + 2))
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)        
        plt.legend(fontsize = 14)
        plt.grid()

    fig.savefig(os.path.join(save_path, figure_title))
    plt.close(fig)
