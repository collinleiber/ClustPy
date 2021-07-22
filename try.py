
# %%
from cluspy.test import generate_syndata, apply_dipext
import numpy as np

data, rotdata, clusters = generate_syndata([[[0,4,0,1,5,0,2],[0,0,0,1,4,0,3],[3,2,4,1,4,0,3],[0,0,2,1,5,2,0]], [[0,0],[0,4]]], [[0.5, 0.3, 0.7, 0.6], [0.1, 0.4]], [[20,20,20,20], [30,50]])
# apply_dipext(data, np.array(clusters), 'syn')
apply_dipext(rotdata, np.array(clusters), 'synrot')


# %%
from cluspy.data import real_world_data
from cluspy.test import apply_dipext

data, clusters = real_world_data.load_cmu_faces()
apply_dipext(data, clusters, 'faces')

# %%
import numpy as np
from unidip import UniDip

dat = np.concatenate([np.random.randn(200)-3, np.random.randn(200)+3])

dat = np.msort(dat)

intervals = UniDip(dat).run()
print(intervals)

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
