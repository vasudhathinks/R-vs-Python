# Using K-Means to cluster different iris plants based on data
# attributes via "iris-headers.txt" file. 

# Clear environment and set working directory to where data is stored
rm(list=ls())
getwd()

# Amend setting the working directory as necessary
setwd("/Users/vasudha/Documents/GitHub/R_vs_Python/K_Means/Iris_data/")
getwd()

# Load the data and see what it is
iris = read.table("iris-headers.txt", header = TRUE)
class(iris)
colnames(iris)
summary(iris)
str(iris)
plot(iris)

# Install/Load necessary packages to visualize data
install.packages("ggplot2")
library(ggplot2)

# Add'l plotting to see aspects of data
colnames(iris)
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) + geom_point()
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point() # More distinguishable

# Create/Run kmeans model with all 4 attributes with 3 centers/clusters
set.seed(456) # Set seed for replicability
kmeans_all_attributes = kmeans(iris[1:4], centers = 3, iter.max = 20, nstart = 20)
kmeans_all_attributes # Within cluster sum of squares / Total sum of squares: 88.4% -- goal is to 
# maximize this value, without accuracy sufferring (tight clusters are better than spread out ones)

table(kmeans_all_attributes$cluster, iris$Species)
# Output
#   setosa versicolor virginica
# 1     50          0         0
# 2      0         48        14
# 3      0          2        36

# Downside of clustering and unsupervised learning is that we have to do some manual work; in this
# case, we have to decide with cluster (the numbers) should be associated to which labeled
# species type. We can estimate accuracy by summing the 3 max values in different cluster 
# assignments: (50 + 48 + 36) / 150 = 89.3%

# Visually compare original clusters to to the model's clusters
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point() # Original
ggplot(iris, aes(Petal.Length, Petal.Width, color = kmeans_all_attributes$cluster)) + geom_point() # Model


# Let's now model with the 2 Petal attributes since they are more distinguishable
set.seed(654)
kmeans_petal = kmeans(iris[3:4], centers = 3, iter.max = 20, nstart = 20)
kmeans_petal # Within cluster sum of squares / Total sum of squares: 94.3% -- much higher than before
table(kmeans_petal$cluster, iris$Species)
# Output: 
#   setosa versicolor virginica
# 1     50          0         0
# 2      0          2        46
# 3      0         48         4

# Similar to before, we can estimate the accuracy by summing the 3 max values in different 
# clusters: (50 + 48 + 46) / 150 = 96% -- also higher than before. 

# Visually compare original clusters to to the model's clusters
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point() # Original
ggplot(iris, aes(Petal.Length, Petal.Width, color = kmeans_petal$cluster)) + geom_point() # Model
# One can see an line (figuratively with the points) in the graph that separates 
# versicolor from virginica


# Let's see if we can increase accuracy and/or cluster tightness percentage by increasing k. 
# (Continuing to use only Petal attributes since performance was much better)

# Create model for number of centers (k) = 4
kmeans_petal_4 = kmeans(iris[3:4], centers = 4, iter.max = 20, nstart = 20)
kmeans_petal_4 # Within Cluster SS / Total SS = 96.5%, higher than before meaning more compact clusters
table(kmeans_petal_4$cluster, iris$Species) 
# Output: 
#     setosa versicolor virginica 
# 1      0          0        35
# 2      0         24        15   # Cluster 2 is notably split with versicolor and virginia --> not ideal
# 3      0         26         0   # To increase accuracy, we will consider cluster 2 with versicolor
# 4     50          0         0
# Accuracy = (50 + 50 + 35) / 150 = 90%, lower than k = 3

ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point() # Original
ggplot(iris, aes(Petal.Length, Petal.Width, color = kmeans_petal_4$cluster)) + geom_point() # Model

# This model exemplifies what it means to balance accuracy and cluster tightness. We saw the latter
# increase with k = 4, but the former decrease. Note that it is expected that clusters will become
# spatially closer/tighter together as k increases because there are fewer points to average distance
# from the centroid over per cluster and more clusters will lead to smaller spaces between points. 


# Another model for number of centers (k) = 5
kmeans_petal_5 = kmeans(iris[3:4], centers = 5, iter.max = 20, nstart = 20)
kmeans_petal_5 # Within Cluster SS = 97.5%, even higher (makes sense as we add more clusters, we will get more compact ones)
table(kmeans_petal_5$cluster, iris$Species) 
# Output: 
#     setosa versicolor virginica 
# 1      0          0        30   # Cluster 1 & 5 together = virginica
# 2     50          0         0
# 3      0         28         7   # Cluster 3 & 4 together = versicolor
# 4      0         22         0
# 5      0          0        13
# Accuracy = (50 + 50 + 43) / 150 = 95.3%, still lower than k = 3! 

ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point() # Original
ggplot(iris, aes(Petal.Length, Petal.Width, color = kmeans_petal_5$cluster)) + geom_point() # Model
