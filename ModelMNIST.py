from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#carga de datos
mnist = fetch_openml('mnist_784', as_frame=False, parser="liac-arff")
X,y = mnist["data"],mnist["target"]

#division de datos
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#entrenamiento
model = LogisticRegression(max_iter=2500, verbose=1)
model.fit(X_train,y_train)

#evaluacion
y_pred = model.predict(X_test)
acc = accuracy_score(y_test,y_pred)
print(f"Accuracy: {acc}")

'''
Se obtuvo Accuracy: 0.9145714285714286
'''

#guardar modelo
joblib.dump(model, 'mnist.pkl')