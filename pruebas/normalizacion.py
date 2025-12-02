# %% STANDARDSCALER

from sklearn.preprocessing import StandardScaler
data = [[0,0], [0,0], [1,1], [1,1]]
scaler = StandardScaler()
print(scaler.fit(data))
print(scaler.mean_)
print(scaler.transform(data))
print(scaler.transform([[2,2]]))
# %% MINMAX SCALER

from sklearn.preprocessing import minmax_scale
X = [[-2,1,2], [-1,0,1]]
minmax_scale(X, axis=0)
minmax_scale(X, axis=1)
# %% NORMALIZE

from sklearn.preprocessing import normalize
X = [[-2, 1, 2], [-1, 0, 1]]
normalize(X, norm="l1")  # L1 normalization each row independently
normalize(X, norm="l2")  # L2 normalization each row independently

# %%