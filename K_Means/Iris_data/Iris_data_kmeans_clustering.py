# Import necessary libraries/packages (may need to be configured based on the environment being used)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix as cm

# Load/Read data and see what it's like
iris = pd.read_table('iris-headers.txt', delim_whitespace=True)
print(iris.head())
print(iris.shape)

# Convert labelled string values to factors/integers (needed for color scheme in plt)
classes = np.array(iris['Species'])
classnames, labels = np.unique(classes, return_inverse=True)
print(labels) # double-check

# Plot some of the features to see distinguishing factors
plt.figure(1)
plt.scatter(iris['Sepal.Length'], iris['Petal.Width'], c=labels) # c would not accept string
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')
plt.show()

plt.figure(2)
plt.scatter(iris['Petal.Length'], iris['Petal.Width'], c=labels)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show() # As seen in R, these 2 factors are more distinguishable

# Save attributes in separate array for ease in later use (we do not want to use labels in model)
iris_attributes = iris.copy()
del iris_attributes['Species']
print(iris_attributes.shape)

# Run KMeans
np.random.seed(123) # Set seed for replicability
kmeans_all_attributes = KMeans(n_clusters=3, init='random').fit(iris_attributes)
kmeans_all_attributes_predictions = kmeans_all_attributes.labels_

# Create confusion matrix of actual and modeled clusters
cm_all_attributes = cm(labels, kmeans_all_attributes_predictions)
print(cm_all_attributes) # Downside of unsupervised learning, need to manually assign clusters
accuracy_all_attributes = (50+36+48) / 150
print(accuracy_all_attributes) # = 89.3% (same as our performance in R)

# Let's also model with the 2 Petal attributes since they are more distinguishable
np.random.seed(456) # Set seed for replicability
kmeans_petal = KMeans(n_clusters=3, init='random').fit(iris_attributes[['Petal.Length', 'Petal.Width']])
kmeans_petal_predictions = kmeans_petal.labels_

# Create confusion matrix of this model
cm_petal = cm(labels, kmeans_petal_predictions)
print(cm_petal) # Downside of unsupervised learning, need to manually assign clusters
accuracy_petal = (46+48+50) / 150
print(accuracy_petal) # = 96.0% (same as our performance in R)

# Compare actual and modeled clusters
fig3 = plt.figure(3)
ax = fig3.add_subplot(111)
ax.scatter(iris['Petal.Length'], iris['Petal.Width'], c=labels, marker='s', label='actual')
ax.scatter(iris['Petal.Length'], iris['Petal.Width'], c=kmeans_petal_predictions, marker='.', label='model')
plt.legend(loc='upper left')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
