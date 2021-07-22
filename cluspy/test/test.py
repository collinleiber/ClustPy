import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cluspy.centroid import DipExt
from scipy.stats import special_ortho_group

def generate_syndata(centroids, scales, sizes):
    rng = np.random.default_rng()
    alldata = np.empty((sum(sizes[0]), 0))
    allclusters = np.empty((sum(sizes[0]), 0))
    for j in range(len(centroids)):
        data = np.empty((0, len(centroids[j][0])))
        clusters = np.empty((0, 1))
        for i in range(len(centroids[j])):
            a = rng.normal(loc=centroids[j][i], scale=scales[j][i], size=(sizes[j][i], len(centroids[j][i])))
            tmp_clusters = np.zeros((a.shape[0], 1)) + i
            clusters = np.concatenate((clusters, tmp_clusters))
            data = np.concatenate((data, a))
        tmp = np.concatenate((data, clusters), axis=1)
        rng.shuffle(tmp)
        alldata = np.concatenate((alldata, tmp[:,:-1]), axis=1)
        allclusters = np.concatenate((allclusters, tmp[:,-1:]), axis=1)
    q = special_ortho_group.rvs(alldata.shape[1])
    rotdata = alldata @ q
    return alldata, rotdata, allclusters

def apply_dipext(data, clusters, name):
    for i in range(clusters.shape[1]):
        draw_scatter(data, clusters[:,i:i+1], name + '_origin_' + str(i))
    dipExt = DipExt()
    subspace = dipExt.fit_transform(data)
    tmp_data = np.empty((subspace.shape[0], subspace.shape[1] + clusters.shape[1]), dtype=object)
    tmp_data[:,:-clusters.shape[1]] = subspace
    tmp_data[:,-clusters.shape[1]:] = clusters
    daf = pd.DataFrame(data)
    daf.to_csv(name + '_origin.csv', index=False, header=False)
    df = pd.DataFrame(tmp_data)
    df.to_csv(name + '_sub.csv', index = False, header = False)
    for i in range(clusters.shape[1]):
        draw_scatter(subspace, clusters[:,i:i+1], name + '_sub_' + str(i))

def draw_scatter(data, clusters, name):
    tmp_data = np.empty((data.shape[0], data.shape[1] + 1), dtype=object)
    tmp_data[:,:-1] = data
    tmp_data[:,-1:] = clusters
    df_columns = ['x' + str(i) for i in range(1, tmp_data.shape[1])]
    df_columns.append('cluster')
    df = pd.DataFrame(tmp_data, columns=df_columns)

    plot = sns.pairplot(df, hue='cluster', diag_kind='hist')
    plot.savefig(name)
    plt.show()
