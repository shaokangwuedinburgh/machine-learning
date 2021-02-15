#!/usr/bin/env python
# coding: utf-8

# In this lab we look at various unsupervised dimensionality reduction methods. By reducing the dimensionality to two or three dimensions only, we can visualise the data by e.g. scatter plots. Moreover, when we deal with labeled data, we may hope that in the low-dimensional space the classes are well-separated, that is, the transformed low-dimensional data form clusters which correspond to the different classes.
# 
# As in Lab 1, we use the landsat satellite dataset which is 36-dimensional and comprises 6 classes.

# In[2]:


# to get rid of possible future warnings cluttering the notebook
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import required packages 
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import random
from sklearn import manifold
get_ipython().run_line_magic('matplotlib', 'inline')


# # Preprocessing and initial visualisations

# ## Question 1: loading the data
# Load the `landsat_train.csv` dataset into a `pandas` DataFrame called  `landsat_train` and display the shape of the DataFrame. Moreover, load the label names stored in `landsat_classes.csv` into a single dictionary called `landsat_labels_dict` and print the names of the classes.

# In[3]:


# Your code goes here

path_base = os.path.join(os.getcwd(), 'datasets', 'landsat')
path_train = os.path.join(path_base, 'landsat_train.csv')
landsat_train = pd.read_csv(path_train, delimiter = ',')
print("There are {} entries and {} columns in the landsat_train DataFrame"      .format(landsat_train.shape[0], landsat_train.shape[1]))

print('\n')
labels_path = os.path.join(path_base, 'landsat_classes.csv')
landsat_labels = pd.read_csv(labels_path, delimiter = ',', index_col=0)
landsat_labels_dict = landsat_labels.to_dict()["Class"]
print(landsat_labels_dict)


# Now we want to replace the label numbers in the `landsat_train` DataFrame with the corresponding class names. We can achieve that by using the `pandas` function `replace()`. The `inplace` argument determines whether the method alters the object it is called upon and returns nothing, or returns a new object (when `inplace` is set to `False`).  
# 
# Execute the cell below which performs this replacement. The second line is used to show a random sample of 5 entries of the DataFrame for us to inspect the outcome of this transformation.

# In[4]:


# Replace label numbers with their names
landsat_train.replace({'label' : landsat_labels_dict}, inplace=True)
landsat_train.sample(n=5, random_state=10)


# Finally, we would like to store the features and the labels in two different `numpy` arrays. For that, will use the following two methods:
# * the `pandas` `drop()` method to remove columns or rows from a DataFrame ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)). We will use it to drop the `label` column.
# * the `to_numpy` method to transform a DataFrame into float64 numpy array ([documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html)).
# 

# In[5]:


X = np.array(landsat_train.drop('label', axis=1)) # Input features
X = X.astype(np.float64)

y = np.array(landsat_train['label'])  # Labels
print('Dimensionality of X: {}\nDimensionality of y: {}'.format(X.shape, y.shape))

print(type(X[0][0]))


# ## Question 2: feature standardisation

# Feature standardisation is a pre-processing technique that is often used to transform data so that the variables/features are standardised to have the same location and scale. For many algorithms, this is a very important step for training models (both in the context of regression and classification). Read about feature standardisation in the lecture notes and e.g. [here](http://scikit-learn.org/stable/modules/preprocessing.html). 
# 
# Scikit-learn offers an [implementation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) of feature standardisation. Machine learning methods in scikit-learn are implemented as `estimators` with a uniform API (see the [documentation](https://scikit-learn.org/stable/developers/develop.html), optional reading). All estimators implement a `fit` method to learn from data. Classes for supervised learning further provide the `predict` method which predicts outputs (labels) from inputs (e.g. the data matrix `X` above). Classes for unsupervised learning provide the `transform` method to transform the original data into an alternative representation. Sometimes, fitting and transforming can be done more efficiently together. In this case, the classes provide the `fit_transform` method.
# 
# Create a `StandardScaler` and use it to fit and transform `X`. Save the results in a new array `X_sc`. Print the means and standard deviations (of the first 4 columns/features) of the original data `X` and the standardised data `X_sc` as a sanity check. 
# 
# **For the rest of this lab you should use the standardised data (i.e. `X_sc`), unless you are explicitly asked to do otherwise.**
# 

# In[6]:


# Your code goes here

# import
from sklearn.preprocessing import StandardScaler

# combined fitting and transforming
X_sc = StandardScaler().fit_transform(X)

cut = 4
print('Mean of the first 4 columns in X: \n{}'.format(X.mean(axis=0)[0:cut]))
print('Mean of the first 4 columns in X_sc: \n{}'.format(X_sc.mean(axis=0)[0:cut]))
print('\n')
print('Standard deviation of the first 4 columns in X: \n{}'.format(X.std(axis=0)[0:cut]))
print('Standard deviation of the first 4 columns in X_sc: \n{}'.format(X_sc.std(axis=0)[0:cut]))


# ## Question 3: visualisations
# In order to get a first idea about the data, we can use basic explorations and visualisations as those in lab 1. Below we will be using various dimensionality reduction methods to obtain (hopefully!) more informative visualisations in 2D. For your convenience, we provide the following function `scatter_2d_label()` to create a 2D scatter plot that also annotates the corresponding classes appropriately. Execute the following cell and make sure you understand what this function does.

# In[7]:


def scatter_2d_label(X_2d, y, ax=None, s=2, alpha=0.5, lw=2):
    """Visualise a 2D embedding with corresponding labels.
    
    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.
    
    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.
        
    ax : matplotlib axes.Axes 
         axes to plot on
         
    s : float
        Marker size for scatter plot.
    
    alpha : float
        Transparency for scatter plot.
        
    lw : float
        Linewidth for scatter plot.
    """
    
    targets = np.unique(y)  # extract unique labels
    colors = sns.color_palette(n_colors=targets.size)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    # scatter plot    
    for color, target in zip(colors, targets):
        ax.scatter(X_2d[y == target, 0], X_2d[y == target, 1], color=color, label=target, s=s, alpha=alpha, lw=lw)
    
    # add legend
    ax.legend(loc='center left', bbox_to_anchor=[1.01, 0.5], scatterpoints=3, frameon=False); # Add a legend outside the plot at specified point
    
    return ax


# The following cell selects two columns of `X_sc` (i.e. features in the high-dimensional space) and uses the `scatter_2d_label()` function provided above to visualise the 2D scatter plots. Feel free to experiment with other dimensions too.

# In[8]:


dim_1 = 19 # First dimension
dim_2 = 25 # Second dimension

fig, ax = plt.subplots()
scatter_2d_label(X_sc[:, [dim_1,dim_2]], y, ax=ax)
ax.set(xlabel='Dim {}'.format(dim_1), ylabel= 'Dim {}'.format(dim_2));


# Plotting dimension 19 versus 25 shows that e.g. cotton crop takes on rather different values than the other classes even after standardisation, and hence should be easy to separate in a classification problem.
# 
# An alternative to scatter plots are two-dimensional kernel density estimates. The function `kde_2d_label()` below produces a two-dimensional kernel density estimates separately for each class. Make sure to understand what each line does. 
# 
# Below we discuss in more detail the choice of the colour palette, i.e. the commands `sns.color_palette` and `sns.dark_palette` as they are used in many more contexts than kernel density estimation. If you would like to know more about the `mlines.Line2D([],[], ...)` command, see [here](https://matplotlib.org/3.3.3/tutorials/intermediate/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists) (optional reading).

# In[9]:


def kde_2d_label(X_2d, y, ax=None):
    """Kernel density estimate in a 2D embedding with corresponding labels.
    
    X_2d : ndarray, shape (n_samples,2)
        Data to plot
    
    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.
        
    ax : matplotlib axes.Axes 
         axes to plot on    
    """
    
    if ax is None:
        fig, ax = plt.subplots()
        
    targets = np.unique(y)
    palette_name = 'bright'
    colors = sns.color_palette(palette_name, n_colors=targets.size)
    lines = []
    for color, target in zip(colors, targets):
        sns.kdeplot(X_2d[y==target, 0], X_2d[y==target, 1], ax=ax, cmap=sns.dark_palette(color, as_cmap=True))
        lines.append(mlines.Line2D([], [], color=color, label=target))  # dummy line for the legend
    
    # add legend
    ax.legend(lines, targets, loc='center left', bbox_to_anchor=[1.01, 0.5], frameon=False) 
    


# In[10]:


# the scatter plot from above becomes
kde_2d_label(X_sc[:, [dim_1,dim_2]], y)
ax = plt.gca()
ax.set(xlabel='Dim {}'.format(dim_1), ylabel= 'Dim {}'.format(dim_2));


# With `sns.color_palette`, we can control the colours used to visualise the data. Have a quick look at its [documentation](https://seaborn.pydata.org/generated/seaborn.color_palette.html). The most important parameters are `palette` and `n_colors`, which specifies how many number of colours of the palette to use. The options for `palette` are a bit more complicated but it is worthwhile to understand the different kinds of palettes that are available. The explanations below are based on the seaborn tutorial ["Choosing color palettes"](https://seaborn.pydata.org/tutorial/color_palettes.html).
# 
# An important guiding principle in choosing palettes is to use different hues ("colors", e.g. red or blue) to represent categories that do not have any ordering (e.g. the class labels above) while using variations in luminance/lightness to represent quantities with a natural ordering (e.g. numbers). Saturation of a colour can be used make different hues look more distinct.
# 
# The available palettes can be classified into three categories:
# - qualitative palettes: these are used for categorical data. An example is `tab10` (slightly more intense than the default palette) or `Set2`.
# - sequential palettes: these are appropriate when data have a natural ordering and range from relatively low or uninteresting values to relatively high or interesting values (or vice versa). An example is `viridis`. Note that every continuous colormap has a reversed version, which has the suffix "_r", e.g. `viridis_r`.  
# - diverging palettes: these are best suited for datasets where both the low and high values are of equal interest while the midpoint should be deemphasised. This can be used to visualise extreme events such as hot and cold temperatures. An example is `vlag`.
# 
# The default palette that you get with `palette=None` is of the qualitative type. Seaborn offers six variations of the default that have different levels of luminance and saturations. You can set them by choosing one of the following values for the `palette` parameter:  `deep`, `muted`, `pastel`, `bright`, `dark`, `colorblind`.  The tutorial has a [helpful figure](https://seaborn.pydata.org/tutorial/color_palettes.html#qualitative-color-palettes) that shows the resulting colormaps along the saturation and luminance "dimensions". [This page](https://gist.github.com/mwaskom/b35f6ebc2d4b340b4f64a4e28e778486) visualises how the different palettes appear for people with various color vision deficiencies. 
# 
# In the code above, we also use `sns.dark_palette(color, as_cmap=True)`. This, and its counterpart `light_palette` can be used to generate custom sequential palettes that start with either dark or light values and smoothly ramp up to the target specified as `color`. This allows us to visualise the values of the kernel density estimate with the same hue that we used to represent the category.
# 
# Finally, note that with `as_cmap` we control the type of the return value: RGB values (`as_cmap=False`) or matplotlib colormaps  (`as_cmap=True`).
# 
# Use the piece of code below to experiment with different colour palettes. You can use `plt.colormaps()` to get a (long) list of matplotlib colourmaps and/or have a look at its [documentation](https://matplotlib.org/api/pyplot_summary.html?highlight=colormaps#matplotlib.pyplot.colormaps). A further great resource for colour palettes is the [Color Brewer](https://colorbrewer2.org/). 

# In[11]:


targets = np.unique(y)

# for example
palette_names = [None, 'tab10', 'dark', 'viridis', 'viridis_r', 'vlag', 'PuOr', 'gnuplot2']

for i, palette in enumerate(palette_names):
    colors = sns.color_palette(palette, n_colors=targets.size)
    sns.palplot(colors)
    plt.title('Palette name: ' + str(palette))


# # Linear dimensionality reduction by PCA

# In Lab 2, we implemented principal component analysis (PCA) from scratch and have seen how to [perform PCA with scikit-learn](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). We will use it here for (linear) dimensionality reduction. 

# ## Question 4: computing the PC scores
# Use scikit-learn to compute the first two PC scores from the (standardised) input data and visualise the lower-dimensional representation with `scatter_2d_label` or `kde_2d_label`. Interpret the learned representation.

# In[12]:


# Your code goes here

from sklearn.decomposition import PCA # Import the PCA module
pca = PCA(n_components=2)  # Initialise a PCA instance
X_pca_2d = pca.fit_transform(X_sc) # fit using X_sc and then transform X_sc

fig, ax = plt.subplots()
scatter_2d_label(X_pca_2d, y, ax)
ax.set(title='Labelled data in 2-D PCA space', 
       xlabel='Principal component score 1',
       ylabel='Principal component score 2');
#ax.legend().loc = 'best'  # if you want to place the legend elsewhere


kde_2d_label(X_pca_2d, y)
ax.set(title='Labelled data in 2-D PCA space', 
       xlabel='Principal component score 1',
       ylabel='Principal component score 2');


# *Your answer goes here*
# 
# The cotton crop is well separated in this two dimensional space, but we could already see that by plotting dimension 19 versus dimension 25. More interesting are the nature of the principal component directions. The first principal component direction can be considered to be related to "dampness": As we move along the x-axis (first PC direction), we move from "very damp grey soil", via "damp grey soil" to "grey soil". The second principal component direction may be related to the type of soil or luminance as we move from "grey" via "red" to "white" (the cotton crop). While PCA has identified meaningful directions, we may want a representation where the different kinds of categories are more separated. 

# # Dimensionality reduction by kernel PCA

# In PCA, the lower dimensional representation is obtained by linear projections and this can severely limit the usefulness of the approach. Several versions of nonlinear PCA have been proposed in the hope of overcoming this problem. One such algorithm is kernel PCA (KPCA).
# 
# Kernel PCA uses the  "kernel trick" to create a nonlinear version of PCA in sample space by performing ordinary PCA in the augmented kernel space. Scikit-learn offers an implementation of KPCA that supports various kernels.  Familiarise yourself with the `KernelPCA` class  by reading the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA).

# ## Question 5: effect of the kernel
# 
# Apply kernel PCA to the standardised data `X_sc`. Set the `n_components` parameter to `2` and use default settings for other parameters. Experiment with different kernels. How do the results differ when different kernels are used?

# In[13]:


from sklearn.decomposition import KernelPCA

# Your code goes here

kernels = ['poly', 'rbf', 'cosine', 'sigmoid']
fig, ax = plt.subplots(2,2,figsize=(12,12));

for ii, kernel in enumerate(kernels):
    X_kpca_2d = KernelPCA(n_components=2, kernel=kernel).fit_transform(X_sc)
    cur_ax = ax[ii//2, ii%2]
    scatter_2d_label(X_kpca_2d, y, ax=cur_ax)
    cur_ax.set(title='{} kernel'.format(kernel))
    cur_ax.legend().set_visible(False)

ax[0, 0].set_ylabel('Principal component 2')
ax[1, 0].set_ylabel('Principal component 2')

ax[1, 0].set_xlabel('Principal component 1')
ax[1, 1].set_xlabel('Principal component 1')

plt.legend(loc='center left', bbox_to_anchor=[1.01, 1.], scatterpoints=3);


# *Your answer goes here*
# 
# The choice of kernel dramatically affects the results of the method. But there are some common patterns (ignoring the not useful representation obtained with the polynomial kernel). In all cases, as in standard PCA above, the grey soils are ordered along a dampness gradient. Moreover, some red soil is more similar to grey soil, and other more to "soil with vegetation stubble", which can be more clearly seen in the representation with the cosine and rbf kernel than with ordinary PCA and the sigmoid kernel.

# ## Question 6: effect of standardisation
# 
# Apply kernel PCA with a RBF kernel to:
# 1. the raw data in `X`
# 2. the standardised data in `X_sc`. 
# 
# What do you observe? Can you explain the outcome?
# 

# In[14]:


# Your code goes here

Z_kpca_sc = KernelPCA(n_components=2, kernel='rbf').fit_transform(X_sc)
Z_kpca = KernelPCA(n_components=2, kernel='rbf').fit_transform(X)

scatter_2d_label(Z_kpca_sc, y)
plt.title('With standardisation')

scatter_2d_label(Z_kpca, y)
plt.title('Without standardisation');


# *Your answer goes here*
# 
# There is big difference between the two representations. The representation without standardisation is not useful at all. This reason is that the rbf kernel uses a common scale for all dimensions ([documentation](https://scikit-learn.org/stable/modules/metrics.html#rbf-kernel)), which is often insufficient if the dimensions have different scales.

# # Multidimensional scaling (MDS)

# Multidimensional scaling (MDS) refers to a collection of techniques for dimensionality reduction that operate on dissimilarities. The goal of MDS is to find a configuration of points in the plane, or more generally the Euclidean space, so that their distances well represent the original dissimilarities.
# 
# We look here into metric MDS (see Section 3.3.1 in the lecture notes). Scikit-learn offers an implementation.  Familiarise yourself with the class `MDS` by reading the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html). Note that with the default settings, Euclidean distances between the input data are used to obtain the pairwise dissimilarities $\delta_{ij}$ from the lecture notes. However, a major strength of MDS is that it operates on dissimilarities directly, and does not require access to data. If you want to operate on dissimilarities, you can to specify `dissimilarity='precomputed'` and feed the corresponding dissimilarity matrix to `fit` or `fit_transform`.
# 
# The attribute `stress_` corresponds to the minimal value of the loss function minimised in MDS (smaller values are better). Note: The stress does not seem to be correctly defined in the [user guide](https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling) (weights and the squaring are missing) but the loss used in the code is the same as the lecture notes (up to a factor 1/2 it seems).

# ## Question 7: MDS based on Euclidean distances
# Compute a two-dimensional representation of the standardised data `X_sc` via scikit-learn using Euclidean distances between the original data points to define the dissimilarities. If the default settings result in too long compute times, you can e.g. reduce `max_iter` to `100` and/or `n_init` to `1`. You can parallelise computations using the `n_jobs` parameter.
# 
# Visualise the data by using the `scatter_2d_label()` function. What do you observe?

# In[18]:


# Your code goes here

from sklearn.manifold import MDS

mds = MDS(n_components=2, n_jobs=-1, random_state=10, max_iter=10)
X_mds_2d = mds.fit_transform(X_sc)

scatter_2d_label(X_mds_2d, y)
plt.title('Metric MDS, stress: {}'.format(mds.stress_))
plt.xlabel('Component 1')
plt.ylabel('Component 2');


# *Your answer goes here*
# 
# The striking difference to other representations is that the cotton crop data are separated into two distinct regions -- one closer to red soil data, and one closer the grey soil data. The previous visualisations placed the cotton data closer to the red soil data. But without further investigation it is unclear whether this is just an artifact to obtain large distances both within and across categories.

# ## Question 8: MDS based on custom distances
# 
# We now apply MDS to a dissimilarity matrix directly. You can define your own dissimilarities to obtain the matrix, or you can also make use of distances that are already implemented in skit-learn or scipy via the `pairwise_distances` function (see [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)).
# 
# Perform here MDS with the `braycurtis` or `canberra` dissimilarity available through `pairwise_distances` and interpret the results. The Bray-Curtis dissimilarity is used in ecology and the environmental sciences (see e.g. [here](http://www.econ.upf.edu/~michael/stanford/maeb5.pdf), optional reading). The Canberra dissimilarity is a weighted $L_1$ distance (see e.g. [here](https://en.wikipedia.org/wiki/Canberra_distance)). For their exact definitions, see [here](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html). 

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

# Your code goes here

my_metrics = ['braycurtis', 'canberra']
for metric in my_metrics:
    delta = pairwise_distances(X=X_sc, metric=metric)
    mds = MDS(n_components=2, dissimilarity='precomputed', n_jobs=-1, max_iter=100)
    X_mds_2d = mds.fit_transform(delta)
    scatter_2d_label(X_mds_2d, y)
    plt.title('Metric MDS ({0}), stress: {1:.2f}'.format(metric, mds.stress_))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2');


# *Your answer goes here*
# 
# The representation is rather different from the one obtained with Euclidean distances, highlighting the importance of choosing a suitable dissimilarity. The representations obtained with the Bray-Curtis and Canberra dissimilarity are quite similar (note that the representations are invariant to rotation of the points because of the loss function is invariant to rotations -- rotating all points by a fixed angle does not change the distances between them). A difference may be that the "soil with vegetation stubble" is more separated from the "red soil" data in case of the Bray-Curtis distance. Overall, the representations are qualitatively similar to those obtain with KPCA (rbf and cosine kernel), and we can again see the variation along the "dampness" and the "type of soil/luminance" dimensions that we have already encountered in PCA.

# #  Isomap
# MDS does not attempt to explicitly model the underlying data manifold. Isomap, on the other hand, addresses the dimensionality reduction problem by doing so. Suppose your data lie on a curve, but the curve is not a straight line. The key assumption made by Isomap is that the quantity of interest, when comparing two points, is the distance along the curve between the two points. 
# 
# In other words, Isomap performs MDS in the geodesic space of the nonlinear data manifold. The geodesic distances represent the shortest paths along the curved surface of the manifold measured as if the surface was flat. This can be approximated by a sequence of short steps between neighbouring sample points. Isomap then applies classical MDS to the geodesic rather than straight line distances to find a low-dimensional mapping that preserves these pairwise distances.
# 
# To summarise, Isomap uses three steps:
# 1.  Find the neighbours of each data point in high-dimensional data
# 2.  Compute the geodesic pairwise distances between all points
# 3.  Embed the data via classical MDS so as to preserve these distances.
# 
# 
# Familiarise yourself with the Isomap class in scikit-learn by reading the [user guide](http://scikit-learn.org/stable/modules/manifold.html#isomap) and  [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html).
# 

# ## Question 9: influence of the neighbourhood
# 
# Project the standardised data into a 2D space via the Isomap algorithm. Explore the role of the `n_neighbors` parameter which defines how many neighbours are used in step 1 above. You can start by trying the following values, but feel free to experiment further: [2, 3, 5, 10]. How sensitive is the algorithm to the choice of `n_neighbors`?
# 
# Use default settings for other parameters.

# In[16]:


from sklearn.manifold import Isomap

# Your code goes here

n_neighbours_arr = [2, 3, 5, 10]
fig, ax = plt.subplots(2,2,figsize=(12,12))

for ii, n_neighbours in enumerate(n_neighbours_arr):
    ismp = Isomap(n_components=2, n_neighbors=n_neighbours)
    X_ismp_2d = ismp.fit_transform(X_sc)
    
    cur_ax = ax[ii//2, ii%2]
    scatter_2d_label(X_ismp_2d, y, ax=cur_ax)
    cur_ax.set(title='{} neighbours'.format(n_neighbours))
    cur_ax.legend().set_visible(False)
 
ax[0, 0].set_ylabel('dimension 2')
ax[1, 0].set_ylabel('dimension 2')

ax[1, 0].set_xlabel('dimension 1')
ax[1, 1].set_xlabel('dimension 1')

plt.legend(loc='center left', bbox_to_anchor=[1.01, 1.], scatterpoints=3);


# *Your answer goes here*
# 
# The parameter `n_neighbors` determines the number of neighbours to consider for each point and affects the construction of the graph that is used to compute the geodesic distance. The representations for 3, 5, and 10 neighbours are quite similar, and also resemble those obtained with PCA. There is a larger difference between the representation for `n_neighbors=2` and `n_neighbors=3`, which indicates that two neighbours may not be sufficient to build the graph reliably.

# #  Other methods for dimensionality reduction [optional reading]
# 
# We here briefly mention two other methods that are widely used and point to code implementing them: 
# 
# * Uniform Manifold Approximation and Projection (UMAP)
# * t-distributed Stochastic Neighbor (t-SNE) Embedding
# 
# UMAP is discussed in the lecture notes. Code that follows the scikit-learn API is available here [here](https://github.com/lmcinnes/umap), and the documentation [here](https://umap-learn.readthedocs.io/en/latest/).
# 
# t-SNE is a powerful tool to visualize high-dimensional data. It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. scikit-learn has an implementation ([documentation](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)). The following webpage provides an excellent interactive introduction to t-SNE that also allows you to see the impact of its different hyperparameters: http://distill.pub/2016/misread-tsne/

# In[ ]:




