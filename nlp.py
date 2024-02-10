import numpy as np
import pandas as pd

#csv dosyasını düzeltme
with open("Restaurant_Reviews.csv","r") as file:
    reviews = []
    liked = []
    for i, line in enumerate(file):
        if i == 0:
            continue
        while not line[-1].isdigit():
            line = line[:-1]
            if line[-1].isdigit():
                    break
        reviews.append(line[:-2])
        liked.append(line[-1])
   
yorumlar = pd.DataFrame(data=zip(reviews, liked), columns=['Reviews', 'Liked'])

#Preprocessing
import re
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download("stopwords")
from nltk.corpus import stopwords

derlem = []
for i in range(1000):
    yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Reviews"][i])
    yorum = yorum.lower()
    yorum = yorum.split()
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]
    yorum = " ".join(yorum)
    derlem.append(yorum)

#Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()#bağımsız değşken
y = yorumlar.iloc[:,1].values #bağımlı değişken

#Makine Öğrenmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm) #%72.5 accuracy