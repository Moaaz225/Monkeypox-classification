#!/usr/bin/env python
# coding: utf-8

# 
# ## The First Trail was to hand the model 
# ## without feature extraction 
# 

# In[2]:


pip install pyradiomics


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import shutil as sh
import cv2 as cv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization , Input
# Instead of 'from keras.preprocessing.image import ImageDataGenerator' use:
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import ImageDataGenerator from tensorflow.keras.preprocessing.image
from tensorflow.keras import regularizers
from keras.applications import ResNet50
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop,Adamax
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
from skimage.feature import graycomatrix, graycoprops , hog 
from radiomics import featureextractor
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


# extract paths from files

# In[4]:


# Augmented Images
os.makedirs('/kaggle/working/data')


# In[5]:


os.makedirs('/kaggle/working/data/Monkeypox')
os.makedirs('/kaggle/working/data/Others')


# In[6]:


data_paths=[]
labels=[]


# In[7]:


for i in os.listdir('/kaggle/input/monkeypox-skin-lesion-dataset/Augmented Images/Augmented Images'):
    for ii in os.listdir('/kaggle/input/monkeypox-skin-lesion-dataset/Augmented Images/Augmented Images'+'/'+i):
        data_paths.append('/kaggle/input/monkeypox-skin-lesion-dataset/Augmented Images/Augmented Images'+'/'+i+'//'+ii)
        labels.append(i)


# In[8]:


for i in os.listdir('/kaggle/input/monkeypox-skin-lesion-dataset/Fold1/Fold1/Fold1'):
    for ii in os.listdir(f'/kaggle/input/monkeypox-skin-lesion-dataset/Fold1/Fold1/Fold1/{i}'):
        for iii in os.listdir(f'/kaggle/input/monkeypox-skin-lesion-dataset/Fold1/Fold1/Fold1/{i}/{ii}'):
            data_paths.append(f'/kaggle/input/monkeypox-skin-lesion-dataset/Fold1/Fold1/Fold1/{i}/{ii}/{iii}')
            labels.append(ii)


# In[9]:


labels[-10:-1]


# In[10]:


for i in os.listdir('/kaggle/input/monkeypox-skin-lesion-dataset/Original Images/Original Images'):
    for ii in os.listdir(f'/kaggle/input/monkeypox-skin-lesion-dataset/Original Images/Original Images/{i}'):
        data_paths.append(f'/kaggle/input/monkeypox-skin-lesion-dataset/Original Images/Original Images/{i}/{ii}')
        labels.append(i)


# In[11]:


len(data_paths)


# In[12]:


df=pd.DataFrame({
    'paths': data_paths,
    'labels': labels
})


# In[13]:


df.head()


# In[14]:


df['labels']=df['labels'].replace('Monkeypox_augmented','Monkeypox')
df['labels']=df['labels'].replace('Others_augmented','Others')
df['labels']=df['labels'].replace('Monkey Pox','Monkeypox')


# In[15]:


for i in range (len(df)):
    img=cv.imread(df.iloc[i,0])
    # print(img)
    img=cv.resize(img,(224,224))
    cv.imwrite(df.iloc[i,0],img)
    cv.imwrite(f'/kaggle/working/data/{df.iloc[i,1]}/{df.iloc[i,0].split("/")[-1]}',img)


# In[16]:


df['labels'].unique()


# In[17]:


len(os.listdir('/kaggle/working/data/Others'))


# In[18]:


len(os.listdir('/kaggle/working/data/Monkeypox'))


# In[19]:


from collections import Counter


# In[20]:


path=[]
label=[]
for i in os.listdir('/kaggle/working/'):
    for ii in os.listdir(f'/kaggle/working/{i}'):
        for iii in os.listdir(f'/kaggle/working/{i}/{ii}'):
            path.append(f'/kaggle/working/{i}/{ii}/{iii}')
            label.append(ii)


# In[21]:


len(label)


# In[22]:


data=[]
for i in path:
    data.append(cv.imread(i))


# In[23]:


data=np.array(data)


# In[24]:


count=Counter(label)
plt.figure(figsize=(10,5))
plt.bar(count.keys(),count.values(),color=['#47f','#dd7'])
plt.xticks(rotation=45)
plt.title('unprocessed_data')
plt.xlabel('Class')


# In[25]:


from imblearn.over_sampling import BorderlineSMOTE , ADASYN


# In[26]:


plt.figure(figsize=(10,10))
c=0
for i in (list(np.random.randint(1000,2500,9))):

    plt.subplot(3,3,c+1)
    img=cv.imread(path[i])
    plt.imshow(img)
    plt.title(f'{label[i]}')
    c+=1


# In[27]:


BL=BorderlineSMOTE()
AS=ADASYN()


# In[28]:


data_resampled , label_resampled = BL.fit_resample(data.reshape(-1,224*224*3),np.array(label))


# In[29]:


proc_count=Counter(label_resampled)


# In[30]:


data_resampled=data_resampled.reshape(-1,224,224,3)


# In[31]:


print(proc_count)
plt.figure(figsize=(10,5))
plt.bar(proc_count.keys(),proc_count.values(),color=['#47f','#dd7'])
plt.xticks(rotation=45)
plt.title('unprocessed_data')
plt.xlabel('Class')


# In[32]:


plt.pie(proc_count.values(),labels=proc_count.keys(),colors=['#47f','#dd7'],autopct='%1.1f%%')
plt.title('Class Distribution')


# In[33]:


dat_re=data_resampled[range(len(data),len(data_resampled))]


# In[34]:


label_re=label_resampled[range(len(label),len(label_resampled))]


# In[35]:


plt.figure(figsize=(10,10))
for i in range (9):
    plt.subplot(3,3,i+1)
    img=dat_re[i]
    plt.imshow(img)
    plt.title(f'{label_re[i]}')


# In[36]:


plt.figure(figsize=(10,10))
for i in range (9):
    plt.subplot(3,3,i+1)
    img=dat_re[i]
    r,g,b=cv.split(img)
    r=cv.equalizeHist(r)
    g=cv.equalizeHist(g)
    b=cv.equalizeHist(b)
    img=cv.merge((r,g,b))
    plt.imshow(img)
    plt.title(f'{label_re[i]}')


# In[37]:


for i in range(len(dat_re)):
    img=dat_re[i]
    r,g,b=cv.split(img)
    r=cv.equalizeHist(r)
    g=cv.equalizeHist(g)
    b=cv.equalizeHist(b)
    img=cv.merge((r,g,b))
    cv.imwrite(f'/kaggle/working/data/{label_re[i]}/augmented_{np.random.randint(10000,50000)}.jpg',img)


# In[38]:


len(os.listdir('/kaggle/working/data/Monkeypox'))


# In[39]:


len(os.listdir('/kaggle/working/data/Others'))


# In[40]:


final_paths=[]
final_labels=[]
for i in os.listdir('/kaggle/working/data'):
    for ii in os.listdir(f'/kaggle/working/data/{i}'):
        final_paths.append(f'/kaggle/working/data/{i}/{ii}')
        final_labels.append(i)


# In[41]:


final_df=pd.DataFrame({'path':final_paths , 'label':final_labels })


# In[42]:


final_df.head()


# In[54]:


import cv2
# from skimage.feature import greycomatrix, greycoprops
import numpy as np

# Load image in grayscale
image_path = r"/kaggle/input/monkeypox-skin-lesion-dataset/Fold1/Fold1/Fold1/Test/Others/NM02_01.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Quantize the image (if needed)
image = (image / 16).astype(np.uint8)  # Reduce to 16 levels

# Compute GLCM
glcm = graycomatrix(image, distances=[1], angles=[0,45], levels=16, symmetric=True, normed=True)

# Extract GLCM features
contrast = graycoprops(glcm, 'contrast')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
dissimilarity =graycoprops(glcm, 'dissimilarity')[0, 0]

# Print features
print(f"Contrast: {contrast}")
print(f"Correlation: {correlation}")
print(f"Energy: {energy}")
print(f"Homogeneity: {homogeneity}")
print(f"dissimilarity: {dissimilarity}")
print(glcm.shape)


# In[55]:


def extract_features_with_grayCoMatrix(img_paths):
    angles=[0,45,90,135]
    features = []
    for img_path in img_paths:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = (img / 16).astype(np.uint8)  # Reduce to 16 levels
        glcm = graycomatrix(img, distances=[1], angles=angles, levels=16, symmetric=True, normed=True)
        feature=[]
        for i in range(len(angles)):
            contrast = graycoprops(glcm, 'contrast')[0, i]
            feature.append(contrast)
            correlation = graycoprops(glcm, 'correlation')[0, i]
            feature.append(correlation)
            energy = graycoprops(glcm, 'energy')[0, i]
            feature.append(energy)
            homogeneity = graycoprops(glcm, 'homogeneity')[0, i]
            feature.append(homogeneity)
            dissimilarity =graycoprops(glcm, 'dissimilarity')[0, i]
            feature.append(dissimilarity)
        features.append(np.array(feature))
    return features


# In[78]:


features = extract_features_with_grayCoMatrix(final_df['path'])
X = np.array(features)
labels={final_df['label'].unique()[0]:0 ,final_df['label'].unique()[1]:1  }
y = final_df['label'].map(labels)


# In[79]:


labels


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[58]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[59]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[60]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[61]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[62]:


features = extract_features_with_grayCoMatrix(df.iloc[:,0])
X = np.array(features)
labels={final_df['label'].unique()[0]:0 ,final_df['label'].unique()[1]:1  }
y = df.iloc[:,1].map(labels)


# re-oversampling with Borderline 

# In[63]:


X_syn , y_syn=BL.fit_resample(X,y)


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)


# In[65]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[66]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[67]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[68]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[ ]:





# re-oversampling with ADASYN 

# In[69]:


ADA=ADASYN()
X_syn , y_syn=ADA.fit_resample(X,y)


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)


# In[71]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[72]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[73]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[74]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[89]:


final_df['path']


# In[98]:


def extract_features_with_hog(img_paths):
    features=[]
    for img_path in img_paths:
        img = cv.imread(f"{img_path}", cv.IMREAD_GRAYSCALE)
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.append(fd)
        print
    return features


# In[99]:


labels


# In[100]:


features = extract_features_with_hog(final_df['path'])


# In[101]:


X=np.array(features)
labels={final_df['label'].unique()[0]:0 ,final_df['label'].unique()[1]:1  }
y = final_df['label'].map(labels)


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[103]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[104]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[105]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[106]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[107]:


features = extract_features_with_hog(df.iloc[:,0])
X = np.array(features)
labels={final_df['label'].unique()[0]:0 ,final_df['label'].unique()[1]:1  }
y = df.iloc[:,1].map(labels)


# re-oversampling with Borderline 

# In[108]:


X_syn , y_syn=BL.fit_resample(X,y)


# In[109]:


X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)


# In[110]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[111]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[112]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[113]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[ ]:





# re-oversampling with ADASYN 

# In[114]:


ADA=ADASYN()
X_syn , y_syn=ADA.fit_resample(X,y)


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)


# In[116]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[117]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[118]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[119]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[ ]:





# In[120]:


def extract_features_with_ResNet(img_paths):
    resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for img_path in img_paths:
        img = cv.imread(img_path)
        image = preprocess_input(np.expand_dims(img, axis=0))
    
    # Extract features
        feature = resnet_model.predict(image)
        features.append(feature.flatten())
    return features


# In[121]:


labels


# In[ ]:


features = extract_features_with_ResNet(final_df['path'])
X = np.array(features)
labels={final_df['label'].unique()[0]:0 ,final_df['label'].unique()[1]:1  }
y = final_df['label'].map(labels)


# In[143]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[144]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[145]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[146]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[147]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[148]:


features = extract_features_with_ResNet(df.iloc[:,0])
X = np.array(features)
labels={final_df['label'].unique()[0]:0 ,final_df['label'].unique()[1]:1  }
y = df.iloc[:,1].map(labels)


# re-oversampling with Borderline 

# In[171]:


X_syn , y_syn=BL.fit_resample(X,y)


# In[172]:


X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)


# In[173]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[174]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[175]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[176]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[ ]:





# re-oversampling with ADASYN 

# In[165]:


ADA=ADASYN()
X_syn , y_syn=ADA.fit_resample(X,y)


# In[166]:


X_train, X_test, y_train, y_test = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn)


# In[167]:


clf = SVC(kernel='poly', C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[168]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[169]:


rf=RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))


# In[170]:


import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm")


# In[ ]:




