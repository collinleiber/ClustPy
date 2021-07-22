#%%
import numpy as np
from dipext import DipExt

data = np.genfromtxt('/home/fridolin/Nextcloud/Uni/Bachi/synData-3-3-1.csv', delimiter=",")
print(data.shape)
tmp = DipExt()
subspace = tmp.transform(data)
print(subspace)
print(tmp.dip_values_)
# %%
from cluspy.test import generate_syndata, apply_dipext
import numpy as np

data, rotdata, clusters = generate_syndata([[[0,4,0,1,5,0,2],[0,0,0,1,4,0,3],[3,2,4,1,4,0,3],[0,0,2,1,5,2,0]], [[0,0],[0,4]]], [[0.5, 0.3, 0.7, 0.6], [0.1, 0.4]], [[20,20,20,20], [30,50]])
# apply_dipext(data, np.array(clusters), 'syn')
apply_dipext(rotdata, np.array(clusters), 'synrot')

# %%
a = np.empty((0, 2))
a = np.concatenate((a, np.zeros((1,2))))
a= np.concatenate((a, np.eye(2)))
print(a.shape, a.shape[0], a.shape[1])
# %%
import numpy as np
from cluspy.test import apply_dipext

data = np.genfromtxt('/home/fridolin/Nextcloud/Uni/Bachi/synData-3-3-1.csv', delimiter=",")
apply_dipext(data[:,2:], data[:,:2], 'synData-3-3-1')
#%%
import numpy as np
from cluspy.test import apply_dipext

data = np.genfromtxt('/home/fridolin/Nextcloud/Uni/Bachi/synData-4-3-2-1.csv', delimiter=",")
apply_dipext(data[:,3:], data[:,:3], 'synData-4-3-2-1')
#%%
import numpy as np
from cluspy.test import apply_dipext

data = np.genfromtxt('/home/fridolin/Nextcloud/Uni/Bachi/synData-5-4-3-1u.csv', delimiter=",")
apply_dipext(data[:,3:], data[:,:3], 'synData-5-4-3-1u')

#%%
a = [1]
a.append([2,3,4])
print(a)
# %%
from PIL import Image
import numpy as np
import os
from test import apply_dipext

directory = './faces_4/'

images = np.empty((624,30*32))
clusters = np.empty((624,4), dtype='<U10')
filenames = os.listdir(directory)
for i in range(len(filenames)):
    images[i:i+1,:] = np.array(Image.open(directory + filenames[i])).flatten()
    clusters[i:i+1,:] = np.array(filenames[i].split('_'))[:4]
apply_dipext(images, clusters, 'faces')

# %%
import pandas as pd
import numpy as np
from test import apply_dipext

df = pd.read_csv('./Absenteeism_at_work_AAA/Absenteeism_at_work.csv', sep=';')
data = df[['Day of the week','Transportation expense','Distance from Residence to Work','Service time','Age','Work load Average/day ','Son','Pet','Weight','Height','Body mass index','Absenteeism time in hours']].to_numpy()
clusters = df[['ID','Reason for absence','Disciplinary failure','Education','Social drinker','Social smoker']].to_numpy()

apply_dipext(data, clusters, 'absenteeism2')


# %%
import pandas as pd
import numpy as np
from test import apply_dipext

df = pd.DataFrame()
for i in [3,6,7,10,12]:
    tmp_df = pd.read_csv('./QCM Sensor Alcohol Dataset/QCM' + str(i) + '.csv', sep=';')
    tmp_df['sensor'] = i
    df = df.append(tmp_df, ignore_index=True)
df['alcohol'] = df['1-Octanol'] + 2*df['1-Propanol'] + 3*df['2-Butanol'] + 4*df['2-propanol'] + 5*df['1-isobutanol']
data = df[['0.799_0.201','0.799_0.201.1','0.700_0.300','0.700_0.300.1','0.600_0.400','0.600_0.400.1','0.501_0.499','0.501_0.499.1','0.400_0.600','0.400_0.600.1','sensor']].to_numpy()
clusters = df[['sensor','alcohol']].to_numpy()

apply_dipext(data, clusters, 'alcohol')
# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from _diptest import dip

data = np.genfromtxt('./synData-4-3-2-1/synData-4-3-2-1_sub.csv', delimiter=',')
# sns.distplot(data[:,0], kde=False)
print(np.sort(data)[2366,0], np.sort(data)[2625,0], np.sort(data)[4895,0])
# %%
import numpy as np
from unidip import UniDip

dat = np.concatenate([np.random.randn(200)-3, np.random.randn(200)+3])

dat = np.msort(dat)

intervals = UniDip(dat).run()
print(intervals)

# %%

a = np.zeros((5,2))
b = np.ones((5,3))

c = np.concatenate((a,b), axis=1)
print(c)

# %%

import numpy as np
from scipy import linalg

rng = np.random.default_rng()
A = rng.standard_normal((4,4))
Q, R = linalg.qr(A)
print(linalg.det(Q), Q)
Q[:,0] = Q[:,0]*(-rng.random() + 1)
print(linalg.det(Q), Q)
Q[:,1] = linalg.det(Q) * Q[:,1]

print(linalg.det(Q), Q)

# %%

import numpy as np
import pandas as pd

df = pd.read_csv('./synrot_origin.csv', header=None)
data = df.to_numpy()
newData = np.concatenate((data, np.zeros((data.shape[0], 1))), axis=1)
newDf = pd.DataFrame(newData)
newDf.to_csv('synrot_origin.csv', index=False, header=False)
# %%

import numpy as np
import pandas as pd
import seaborn as sns
from test import draw_scatter

df = pd.read_csv('./synrot_origin_DipExt.csv', header=None)
subspace = df.to_numpy()

cf = pd.read_csv('./synrot_sub.csv', header=None)
clusters = cf.to_numpy()[1:,-2:]

for i in range(clusters.shape[1]):
    draw_scatter(subspace, clusters[:,i:i+1], 'synrot_DipExt_' + str(i))
# %%
