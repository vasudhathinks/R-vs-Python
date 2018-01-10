# Using SVM to classify whether applicants were successful
# in acquiring a loan via "credit_card_data-headers.txt" file. 

# Clear environment and set working directory to where data is stored
rm(list=ls())
getwd()

# Amend setting the working directory as necessary
setwd("...")
getwd()

# Load the data and see what it is
data = read.table("credit_card_data-headers.txt", header = TRUE)
str(data)
summary(data)
head(data)

# Visualize the data
hist(data$R1) # Specifically the response variable
plot(data$A2, data$A3) # Just to gauge, but we are not given
# feature information so we can't go much further

# Install/Load necessary packages
install.packages("caTools") # For splitting data to train and test sets
install.packages("kernlab") # For SVM model
library(caTools)
library(kernlab)

# Create train and test sets with 75% in training and 25% in test
set.seed(123) # Set seed for replicability
split_bools = sample.split(data$R1, Split = .75)
summary(split_bools)
train = subset(data, split_bools == TRUE)
test = subset(data, split_bools == FALSE)

# The ksvm package in R requires a matrix input
class(train) # Need to convert dataframe to matrix
train = as.matrix(train)
class(train) # Double-check
test = as.matrix(test)
class(test)

# Create SVM model 
# There are a number of hyperparameters one can tweak, a few kernels have been tried.
svm_vanilla = ksvm(train[,1:10], train[,11], scaled=TRUE, kernel="vanilladot", type="C-svc", C=100)

# Let's see what the accuracy of this model is on the training set
train_preditions_vanilla <- predict(svm_vanilla, train[,1:10])
train_cm_vanilla = table(train[,11], train_preditions_vanilla) # cm = confusion matrix
train_accuracy_vanilla = sum(train_preditions_vanilla == train[,11]) / nrow(train)
train_accuracy_vanilla # Accuracy on training set = 0.8693878

# Now, let's check what true accuracy is like by testing model on the test set
test_preditions_vanilla <- predict(svm_vanilla, test[,1:10])
test_cm_vanilla = table(test[,11], test_preditions_vanilla) # cm = confusion matrix
test_cm_vanilla
test_accuracy_vanilla = sum(test_preditions_vanilla == test[,11]) / nrow(test)
test_accuracy_vanilla # Accuracy on the test set = 0.8414634, which is likely to be closer to the
# true accuracy. 


# Let's try a non-linear kernal
svm_poly = ksvm(train[,1:10], train[,11], scaled=TRUE, kernel="polydot", type="C-svc", C=100)

# Let's see what the accuracy of this model is on the training set
train_preditions_poly <- predict(svm_poly, train[,1:10])
sum(train_preditions_poly == train[,11]) / nrow(train) 
# Accuracy on training set = 0.8693878 --> the same as the linear kernal

# Let's also check test set accuracy
test_preditions_poly <- predict(svm_poly, test[,1:10])
sum(test_preditions_poly == test[,11]) / nrow(test) 
# Accuracy on the test set = 0.8414634 --> also the same as linear


# Finally, let's try life's favorite kernal, Gaussian
svm_gauss = ksvm(train[,1:10], train[,11], scaled=TRUE, kernel="rbfdot", type="C-svc", C=100)

# Let's see what the accuracy of this model is on the training set
train_preditions_gauss <- predict(svm_gauss, train[,1:10])
sum(train_preditions_gauss == train[,11]) / nrow(train) 
# Accuracy on training set = 0.9632653 --> that is significantly higher

# Let's also check test set accuracy
test_preditions_gauss <- predict(svm_gauss, test[,1:10])
sum(test_preditions_gauss == test[,11]) / nrow(test) 
# Accuracy on the test set = 0.8109756 --> and this is somewhat lower, I think we can safely
# say that the Gaussian kernal is overfitting the training data. 
