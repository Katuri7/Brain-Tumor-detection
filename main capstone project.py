#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA


# In[67]:


pip install opencv-python


# In[68]:


import os
import cv2

# Define classes dictionary
classes = {'no_tumor':0, 'pituitary_tumor':1}

# Load images and labels
X = []
Y = []
for cls in classes:
    pth = r'C:\Users\RAMBABU\Downloads\dataset5\Training\\' + cls
    if not os.path.exists(pth):
        print(f"Error: Path '{pth}' does not exist.")
    for j in os.listdir(pth):
        img = cv2.imread(os.path.join(pth, j), 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


# In[36]:


import os
path = os.listdir(r'C:\Users\RAMBABU\Downloads\dataset5\Training')
classes = {'no_tumor':0, 'pituitary_tumor':1}


# In[37]:


import numpy as np

X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)

np.unique(Y)


# In[38]:


import pandas as pd
pd.Series(Y).value_counts()

X.shape, X_updated.shape


# In[39]:


import matplotlib.pyplot as plt

plt.imshow(X[0], cmap='gray')


# In[40]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                              test_size=.20)


# In[43]:



xtrain.shape, xtest.shape


# In[44]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[45]:


from sklearn.decomposition import PCA

print(xtrain.shape, xtest.shape)


# In[69]:


pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest


# In[70]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[71]:


import warnings
warnings.filterwarnings('ignore')


# In[72]:


lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)
y_pred_lg = lg.predict(xtest)
y_pred = lg.predict(xtest)


# In[73]:


sv = SVC()
sv.fit(xtrain, ytrain)
y_pred_sv = sv.predict(xtest)
y_pred = sv.predict(xtest)


# In[74]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)
y_pred_knn = knn.predict(xtest)
y_pred = knn.predict(xtest)


# In[75]:


print("Logistic Regression")
print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))
print("Precision:", precision_score(ytest, y_pred_lg))
print("Recall:", recall_score(ytest, y_pred_lg))
print("F1 Score:", f1_score(ytest, y_pred_lg))
print("Confusion Matrix:\n", confusion_matrix(ytest, y_pred_lg))


acc = accuracy_score(ytest, y_pred)

print("Accuracy Score:", acc)


# In[53]:


print("\nSupport Vector Machine")
print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))
print("Precision:", precision_score(ytest, y_pred_sv))
print("Recall:", recall_score(ytest, y_pred_sv))
print("F1 Score:", f1_score(ytest, y_pred_sv))
print("Confusion Matrix:\n", confusion_matrix(ytest, y_pred_sv))

acc = accuracy_score(ytest, y_pred)

print("Accuracy Score:", acc)


# In[54]:


print("\nK-Nearest Neighbors")
print("Training Score:", knn.score(xtrain, ytrain))
print("Testing Score:", knn.score(xtest, ytest))
print("Precision:", precision_score(ytest, y_pred_knn))
print("Recall:", recall_score(ytest, y_pred_knn))
print("F1 Score:", f1_score(ytest, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(ytest, y_pred_knn))

acc = accuracy_score(ytest, y_pred)

print("Accuracy Score:", acc)


# In[55]:


import matplotlib.pyplot as plt

models = ['Logistic Regression', 'SVM', 'KNN']
training_scores = [lg.score(xtrain, ytrain), sv.score(xtrain, ytrain), knn.score(xtrain, ytrain)]
testing_scores = [lg.score(xtest, ytest), sv.score(xtest, ytest), knn.score(xtest, ytest)]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_scores, width, label='Training')
rects2 = ax.bar(x + width/2, testing_scores, width, label='Testing')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Scores by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

plt.show()


# In[56]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Fit the logistic regression model
lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)

# Predict probabilities for the test set
probas = lg.predict_proba(xtest)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(ytest, probas[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[57]:


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC for SVM
fpr_svm, tpr_svm, thresholds_svm = roc_curve(ytest, sv.decision_function(xtest))
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC curve for SVM
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label='SVM (AUC = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()


# In[58]:


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC for KNN
fpr_knn, tpr_knn, thresholds_knn = roc_curve(ytest, knn.predict_proba(xtest)[:,1])
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curve for KNN
plt.plot(fpr_knn, tpr_knn, color='darkorange', lw=2, label='KNN (AUC = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.show()


# In[59]:


pred = knn.predict(xtest)


# In[60]:


misclassified=np.where(ytest!=pred)
misclassified


# In[61]:


print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])


# In[62]:


dec = {0:'No Tumor', 1:'Positive Tumor'}


# In[63]:


plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/RAMBABU/Downloads/dataset5/Testing/')
c=1
for i in os.listdir('C:/Users/RAMBABU/Downloads/dataset5/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('C:/Users/RAMBABU/Downloads/dataset5/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[64]:


plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/RAMBABU/Downloads/dataset5/Testing/')
c=1
for i in os.listdir('C:/Users/RAMBABU/Downloads/dataset5/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('C:/Users/RAMBABU/Downloads/dataset5/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[ ]:





# In[ ]:





# In[ ]:




