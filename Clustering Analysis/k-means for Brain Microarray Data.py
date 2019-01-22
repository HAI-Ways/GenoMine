import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

filePath='C:/your folder/'

# number of samples that will be analyzed in detail
sample_n =30
index_n=sample_n+1   # for loop use only

# find outlier samples
def find_outlier(mean_each_sample):    
    mean_all_smaples = np.mean(mean_each_sample)
    std_all_samples =np.std(mean_each_sample)    
    
    for idx, val in mean_each_sample.iteritems():
        z_score= (val - mean_all_smaples)/std_all_samples 
        if np.abs(z_score) > 3:
            outlier_samples[idx]=val
    return outlier_samples
    
# scale back values obtained from model training
def scale_back(scaler,centroids):
    a=centroids.reshape()
    b=scaler.inverse_transform(a)
    return c    
    
# generate column names (probe plus 946 samples)
cols=['probe']
for i in range(946):
    x=i+1
    colnm='sampl'+str(x)
    cols.append(colnm)
    
# normalized human brain microarray gene expression data H0351.2001 (58692 probes * 946 samples)
# were downloaded from http://human.brain-map.org/static/download
data=pd.read_csv(filePath+"MicroarrayExpression.csv",names=cols,index_col='probe',sep=',')
probes=pd.read_csv(filePath+"Probes.csv",index_col='probe_id',sep=',')
data.head()

# calculate the mean of each sample
mean_each_sample=data.mean(axis=0)
outlier_samples={}
outliers=list(find_outlier(mean_each_sample).keys())

# drop outlier samples
data.drop(data[outliers], axis=1, inplace=True)

# check data distribution for small chunk of data
bigChunk, smallChunk= train_test_split(data.iloc[:,1:10], test_size=0.01, random_state=7)
scatter_matrix(smallChunk)
plt.show()

# calculate the average gene expression level and variance among selected samples
data['mean']=data.mean(axis=1)
data['stdv']=data.std(axis=1)

plt.figure(figsize=(10,5))
plt.plot(data['mean'],data['stdv'],'*')
plt.xlabel("mean")
plt.ylabel("stdv")
plt.show()

# Filter genes that have background level of expression
data=data.loc[data['mean']>2.5]

# choose first 30 samples only for clustering analysis
brain=data.iloc[:,:sample_n]
brainX=brain.values

# data normalization for model training
#so values obtained from models need scale back
scaler=StandardScaler()
scaler=scaler.fit(brainX)
normalized_data=scaler.transform(brainX)

# for k-means model tuning
# when sum square error value(SSE) doesn't go down much furthur, we can choose that number to set n_clusters
clusters_n=[]
sum_square_error = [] 
for i in range(1, 16):
    k_model = KMeans(n_clusters=i, max_iter=1000).fit(normalized_data)
    clusters_n.append(i)
    sum_square_error.append(k_model.inertia_)

# plotting the SSE curve
plt.figure(figsize=(10,5))
plt.plot(clusters_n,sum_square_error)
plt.xlabel("n-cluster")
plt.ylabel("sum square error")
plt.title("Parameter Tuning of K-means")
plt.show()

# In this case, plotting showed 4 clusters may be a better choice
optimized_model = KMeans(n_clusters=4, random_state=0)
optimized_model.fit(normalized_data)
result_cluster=optimized_model.labels_
centroids = optimized_model.cluster_centers_
centroids=scaler.inverse_transform(centroids)  #scale back

# plotting the cluster centroid values again each sample
plt.figure(figsize=(10,5))
plt.plot(range(1,index_n),centroids[0],'g')
plt.plot(range(1,index_n),centroids[1],'b')
plt.plot(range(1,index_n),centroids[2],'y')
plt.plot(range(1,index_n),centroids[3],'r')

# if the samples are for time series or treatment analysis
# then we may see different patterns
plt.xlabel("samples") 
plt.ylabel("expression")
plt.title('Centroid Distribution among Samples')
plt.show()

# add resulted clusters back to original data
brain.loc[:,'cluster']=result_cluster

# scatter plot of cluster centroidS using sample 1 vs sample 2
plt.figure(figsize=(10,5))
plt.plot(brain['sampl1'],brain['sampl2'],'*')
plt.plot(centroids[:,:2][:,0],centroids[:,:2][:,1],'r*')
plt.xlabel("sample 1")
plt.ylabel("sample 2")
plt.title("Centroid of 4 Clusters")
plt.show()

# calculate the correlation values between each gene and the centroid of cluster 4
# high correlation to the centroids may imply gene co-regulation
clust4=brain.loc[brain['cluster']==3]
clust4C=clust4.copy()
centroid4=centroids[3:,:].reshape(sample_n)
clust4T=clust4.iloc[:,:-1].T
clust4T['Centroid']=centroid4
correlation={}
for i in range(clust4T.shape[1]-1):
    a=clust4T.iloc[:,i].corr(clust4T['Centroid'])
    correlation[clust4T.iloc[:1,i:i+1].columns[0]]=a
clust4C['Correlation']=list(correlation.values())

# prepare data for heatmap plotting(top100 and bottom 100 correlation)
clust4C.sort_values(by=['Correlation'],ascending=False,inplace=True)
clust4Ctop100=clust4C.iloc[:100,:]
clust4Cbottom100=clust4C.iloc[-100:,:]
transposed_dataT100=clust4Ctop100.T
data_for_plotT100=transposed_dataT100[:sample_n]
colormapT100=transposed_dataT100[index_n:]
transposed_dataB100=clust4Cbottom100.T
data_for_plotB100=transposed_dataB100[:sample_n]
colormapB100=transposed_dataB100[index_n:]

# heatmap for correlation among cluster 4 genes
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(111)
cmap = plt.cm.rainbow
normalizer = mp.colors.Normalize(vmin=-1.0, vmax=1.0)
scalermap = plt.cm.ScalarMappable(cmap=cmap, norm=normalizer)
for i in range(data_for_plotB100.shape[1]):
    color=colormapB100.iloc[:,i:i+1].values.reshape(1)
    data_for_plotB100.iloc[:,i:i+1].plot(ax=ax1,kind='line',color=cmap(color))  
for i in range(data_for_plotT100.shape[1]):
    color=colormapT100.iloc[:,i:i+1].values.reshape(1)
    data_for_plotT100.iloc[:,i:i+1].plot(ax=ax1,kind='line',color=cmap(color),alpha=0.5)   
centroid4=pd.DataFrame(clust4T['Centroid'].values,clust4.columns[:sample_n])    
centroid4.plot(ax=ax1,color ='red',linewidth=2)    
scalermap.set_array([])
fig.colorbar(scalermap)
plt.xlabel("samples")
plt.ylabel("expression level")
ax1.get_legend().remove()
ax1.set_ylim([0,10])
ax1.set_title('Correlation Expression of Cluster 4 Genes')
plt.show()

#  get gene names by join two files
gene_list=pd.merge(probes, clust4C, left_index=True, right_index=True)
gene_list.loc[:,['probe_name','gene_symbol','gene_name','chromosome','cluster','Correlation']].sort_values(by=['Correlation'],ascending=False,inplace=True)

# export reports to csv files
# clusters and correlation values are included
brain.to_csv(filePath+'clustered_brain_microarray.csv', index=False)
clust4C.to_csv(filePath+'cluster4_correlation.csv', index=False)
