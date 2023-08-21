import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor



sns.set(color_codes=True)


#3. Describe the dataset attributes 

heart_failure = pd.read_csv('heart_failure.csv')

print(heart_failure.describe(include='all'))


#4. Apply data cleaning methods (if needed) 

print(heart_failure.isnull().sum())

heart_failure= heart_failure.dropna()

print(heart_failure.count())


# 5. Apply different data visualization charts min 2 charts 
# 6. Mention your observations on your visualizations 

corrmat = heart_failure.corr()
plt.subplots(figsize=(20,20))
sns.heatmap(corrmat,cmap= "BrBG",annot=True, square=True)
plt.show()



"""
"time" is the most important feature as it would've been very crucial to get diagnosed early with cardivascular issue so as to get timely treatment thus, 
reducing the chances of any fatality. (Evident from the inverse relationship)

"serum_creatinine" is the next important feature as serum's (essential component of blood) abundancy in blood makes it easier for heart to function.

"ejection_fraction" has also significant influence on target variable which is expected since it is basically the efficiency of the heart.

Can be seen from the inverse relation pattern that heart's functioning declines with ageing.
"""

cols= ["#CD5C5C","#FF0000"]
plt.figure(figsize=(5,5))
HBP_distribution=sns.countplot(x=heart_failure['high_blood_pressure'],data= heart_failure, hue ="DEATH_EVENT",palette = cols)
HBP_distribution.set_title("Distribution Of HBP", color="#774571")
plt.show()

"""
the percentage of living people who didnt have high blood pressure is biggest than percentage of living people who have high blood pressure
"""



# 7. Apply anomaly detection technique to find outliers

col_names = list(heart_failure.columns)
std_scaler = StandardScaler()
heart_failure_scaled= std_scaler.fit_transform(heart_failure)
heart_failure_scaled = pd.DataFrame(heart_failure_scaled, columns=col_names)

#plt.boxplot(heart_failure_scaled,labels=col_names,vert=False)
fig = plt.figure(figsize =(20, 20))
plt.show()





#8. Apply predictive analytic techniques min 2 types

#first type LinearRegression
print();print();print();
print('first predictive analytic techniques Linear Regression')
x=heart_failure[['age']].values
y=heart_failure['serum_creatinine']
plt.scatter(x,y,color='coral')
model=LinearRegression(fit_intercept=False)
clf=model.fit(x,y)
print("Coefficient: ",clf.coef_)
predictions = model.predict(x)
for idx in range(len(predictions)):
    print('Actual: ',y[idx],' predicted: ',predictions[idx],' Age: ',x[idx,0])
plt.plot(x,predictions)
plt.xlabel("Age")
plt.ylabel("predicted Serum creatinine")
plt.title("Linear Regression")
plt.show()



#second type DecisionTree

#train 80% & test 20%
print();print();print();
print('second predictive analytic techniques Decision Tree');print()
x = heart_failure[['age', 'anaemia', 'creatinine_phosphokinase','diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine','serum_sodium', 'sex', 'smoking', 'time']].values
y = heart_failure['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
df=pd.DataFrame({'Actual ': y_test , ' Predicted ': y_pred , 'Age ':x[y_test,0]})
print(df);print()
print(classification_report(y_test, y_pred))


"""
#train & test of all of data
model = tree.DecisionTreeRegressor()
model.fit(x,y)
predictions = model.predict(x)

print(model.feature_importances_)

for index in range(len(predictions)):
  print('Actual: ', y[index], 'Prtdicttd :	', predictions[index],' Age: ',x[index,0])
"""







#9. Apply text mining for the text dataset.

text_file = open('text_document.txt',encoding='utf8')

for line in text_file: 
    print(line)
    input_str = line.lower()
    print(input_str)
    input_str = input_str.translate(str.maketrans('','', string.punctuation))
    print(input_str)
    input_str = word_tokenize(input_str)
    print(input_str)
    stop_words = set(stopwords.words('english'))
    input_str = [i for i in input_str if not i in stop_words]
    print(input_str)
    for word in input_str:
      result = input_str.index(word)
      stemmer= PorterStemmer()
      input_str[result] =stemmer.stem(word)
    print(input_str)     
    for word in input_str:
      result = input_str.index(word)
      lemmatizer=WordNetLemmatizer()
      input_str[result] =lemmatizer.lemmatize(word)
    print(input_str)
    sia = SentimentIntensityAnalyzer()
    print(sia.polarity_scores(line))
   
text_file.close()
