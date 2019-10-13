import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("credit card.csv")
data.tail(10)
data.shape
data.isnull().sum
number_records_fraud = len(data[data.Class !=1])
number_records_fraud
number_records_fraud = len(data[data.Class ==1])
number_records_fraud
count_classes = data['Class'].value_counts().plot(kind = 'bar')
plt.title('fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
fraud_indices = np.array(data[data.Class == 1].index)
fraud_indices
data['Amount']
from sklearn.preprocessing import StandardScaler
data['Amount1'] = StandardScaler().fit_transform(data[['Amount']])
data.head()
from sklearn.model_selection import train_test_split
x1=data.drop('Class',axis=1)
y1=data['Class']
data.head(10)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=4)
from sklearn.naive_bayes import GaussianNB
model1=GaussianNB()
model1.fit(x_train,y_train)
pred = model1.predict(x_test)
pred
confusion_matrix=pd.crosstab(y_test,pred,rownames=['Actual'],colnames=['prediction'])
confusion_matrix
from sklearn.metrics import classification_report
report=classification_report(y_test,pred)
print(report)
