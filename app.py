from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image,ImageDraw
import numpy as np
import joblib



app = FastAPI()
#cargar modelo MNIST Se obtuvo Accuracy: 0.9145714285714286
mnist = joblib.load("mnist.pkl")
mnistTest = joblib.load("mnistTest.pkl")

@app.get("/")
def hello_world():
    return "Hello World"


@app.post("/Digitos")
async def predict_mnist(img: UploadFile = File(...)):
    img=Image.open(img.file)
    img=img.resize((28,28))
    img=img.convert("L")
    img=np.array(img)
    img=img.flatten()
    img=img.reshape(1,-1)
    resultado = mnist.predict(img) #se hara la prediccion del numero
    '''
    El predict regresa una lista de valores donde el primer valor es el la prediccion por lo tanto
    se mostrara ese valor
    '''
    return{"Tu numero es": int(resultado[0])} 

@app.post("/DigitosTEST")
async def predict_mnistTEST(img: UploadFile = File(...)):
    img=Image.open(img.file)
    img=img.resize((8,8))
    img=img.convert("L")
    img=np.array(img)
    img=img.flatten()
    img=img.reshape(1,-1)
    resultado = mnistTest.predict(img) #se hara la prediccion del numero
    '''
    El predict regresa una lista de valores donde el primer valor es el la prediccion por lo tanto
    se mostrara ese valor
    '''
    return{"Tu numero es": int(resultado[0])} 