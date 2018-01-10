# Import necessary libraries/packages (may need to be configured based on the environment being used)
import pandas as pd
import matplotlib.pyplot as plt


# Load/Read data and see what it's like
iris = pd.read_table('iris-headers.txt', delim_whitespace=True)
print(iris.head())
print(iris.shape)


# # plt.figure(2, figsize=(8, 6))
# # plt.clf()
# #
# # Plot the training points
# plt.scatter(iris[:, 0], iris[:, 1], c=iris[:, 4])
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# # plt.xlim(x_min, x_max)
# # plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())

plt.plot(iris['Petal.Length'], iris['Petal.Width'], 'o', color=iris['Species'])
plt.axis([1, 2, 0, 1])
plt.show()