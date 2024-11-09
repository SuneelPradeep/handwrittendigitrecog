# import sys
# import numpy as np
# from joblib import dump,load
# import pickle
# from sklearn import model_selection, svm, preprocessing
# from sklearn.metrics import accuracy_score,confusion_matrix
# # from MNIST_Dataset_Loader.mnist_loader import MNIST
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
# from matplotlib import style
# style.use('ggplot')


# # Save all the Print Statements in a Log file.
# old_stdout = sys.stdout
# log_file = open("summary.log","w")
# sys.stdout = log_file

# # Load MNIST Data
# # print('\nLoading MNIST Data...')
# # data = MNIST('./MNIST_Dataset_Loader/dataset/')

# print('\nLoading Training Data...')
# # img_train, labels_train = data.load_training()
# # train_img = np.array(img_train)
# # train_labels = np.array(labels_train)

# # print('\nLoading Testing Data...')
# # img_test, labels_test = data.load_testing()
# # test_img = np.array(img_test)
# # test_labels = np.array(labels_test)


# #Features
# # X = train_img

# # #Labels
# # y = train_labels

# # Prepare Classifier Training and Testing Data
# print('\nPreparing Classifier Training and Validation Data...')
# # X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.1)
# print('\nLoading MNIST Data from Keras...')
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(X_train.reshape[0], -1) / 255.0
# X_test = X_test.reshape(X_test.reshape[0], -1) / 255.0 

#  X_train, X_val, y_train, y_val = model_selection.train_test_split(X,y,test_size=0.1)


# # Pickle the Classifier for Future Use
# print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
# print('\nPickling the Classifier for Future Use...')
# clf = svm.SVC(gamma=0.1, kernel='poly')
# clf.fit(X_train,y_train)

# with open('Digggittt_HandWritten_Digit_Recognition','wb') as f:
# 	pickle.dump(clf, f)

# pickle_in = open('Digggittt_HandWritten_Digit_Recognition','rb')
# clf = pickle.load(pickle_in)
# # dump(clf, 'Digggittt.joblib')
# # clf = load('Digggittt.joblib')

# print('\nCalculating Accuracy of trained Classifier...')
# acc = clf.score(X_test,y_test)

# print('\nMaking Predictions on Validation Data...')
# y_pred = clf.predict(X_test)

# print('\nCalculating Accuracy of Predictions...')
# accuracy = accuracy_score(y_test, y_pred)

# print('\nCreating Confusion Matrix...')
# conf_mat = confusion_matrix(y_test,y_pred)

# print('\nSVM Trained Classifier Accuracy: ',acc)
# print('\nPredicted Values: ',y_pred)
# print('\nAccuracy of Classifier on Validation Images: ',accuracy)
# print('\nConfusion Matrix: \n',conf_mat)

# # Plot Confusion Matrix Data as a Matrix
# plt.matshow(conf_mat)
# plt.title('Confusion Matrix for Validation Data')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.savefig('confusion_matrix_validation.png')
# # plt.show()


# print('\nMaking Predictions on Test Input Images...')
# test_labels_pred = clf.predict(test_img)

# print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
# acc = accuracy_score(test_labels,test_labels_pred)

# print('\n Creating Confusion Matrix for Test Data...')
# conf_mat_test = confusion_matrix(test_labels,test_labels_pred)

# print('\nPredicted Labels for Test Images: ',test_labels_pred)
# print('\nAccuracy of Classifier on Test Images: ',acc)
# print('\nConfusion Matrix for Test Data: \n',conf_mat_test)

# # Plot Confusion Matrix for Test Data
# plt.matshow(conf_mat_test)
# plt.title('Confusion Matrix for Test Data')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.axis('off')
# plt.savefig('confusion_matrix_Test_data_validation.png')
# # plt.show()

# sys.stdout = old_stdout
# log_file.close()


# # Show the Test Images with Original and Predicted Labels
# a = np.random.randint(1,40,15)
# for i in a:
# 	two_d = (np.reshape(test_img[i], (28, 28)) * 255).astype(np.uint8)
# 	plt.title('Original Label: {0}  Predicted Label: {1}'.format(test_labels[i],test_labels_pred[i]))
# 	plt.imshow(two_d, interpolation='nearest',cmap='gray')
# 	plt.savefig('PredictedLabel.png')
# 	# plt.show()
# #---------------------- EOC ---------------------#


import sys
import numpy as np
import pickle
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from keras.datasets import mnist  # Import Keras MNIST

# Save all the Print Statements in a Log file.
# old_stdout = sys.stdout
# log_file = open("summary.log", "w")
# sys.stdout = log_file

# Load MNIST Data from Keras
print('\nLoading MNIST Data from Keras...')
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess Data: Flatten the images
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Normalize to [0, 1]
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

# Prepare Classifier Training and Testing Data
print('\nPreparing Classifier Training and Validation Data...')
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.1)

# Pickle the Classifier for Future Use
print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
print('\nPickling the Classifier for Future Use...')
clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(X_train, y_train)

with open('HandWritten_Digit_Recognition', 'wb') as f:
    pickle.dump(clf, f)

# Load the pickled classifier for validation
pickle_in = open('HandWritten_Digit_Recognition', 'rb')
clf = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_val, y_val)

print('\nMaking Predictions on Validation Data...')
y_pred = clf.predict(X_val)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(y_val, y_pred)

print('\nCreating Confusion Matrix...')
conf_mat = confusion_matrix(y_val, y_pred)

print('\nSVM Trained Classifier Accuracy: ', acc)
print('\nPredicted Values: ', y_pred)
print('\nAccuracy of Classifier on Validation Images: ', accuracy)
print('\nConfusion Matrix: \n', conf_mat)

# Plot Confusion Matrix Data as a Matrix
plt.matshow(conf_mat)
plt.title('Confusion Matrix for Validation Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('newconfusion_matrix_validation.png')

# Making Predictions on Test Input Images
y_test_pred = clf.predict(X_test)

print('\nCalculating Accuracy of Trained Classifier on Test Data... ')
test_acc = accuracy_score(y_test, y_test_pred)

print('\nCreating Confusion Matrix for Test Data...')
conf_mat_test = confusion_matrix(y_test, y_test_pred)

print('\nPredicted Labels for Test Images: ', y_test_pred)
print('\nAccuracy of Classifier on Test Images: ', test_acc)
print('\nConfusion Matrix for Test Data: \n', conf_mat_test)

# Plot Confusion Matrix for Test Data
plt.matshow(conf_mat_test)
plt.title('Confusion Matrix for Test Data')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.axis('off')
plt.savefig('newconfusion_matrix_Test_data_validation.png')

sys.stdout = old_stdout
log_file.close()

