
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from shark_id.predict import load_model, predict_image_model

#iniate api
app = FastAPI()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allows all methods
    allow_headers=['*'],  # Allows all headers
)

# preload the model
app.state.model = load_model()



# predicts from url provided by user
#@app.get('/predict_url')
#def prediction(url_with_pic, model_type=''):
#    model = app.state.model
#    assert model is not None
#    prediction = predict.predict_labels(model, model_type, url_with_pic=url_with_pic)
#    return prediction


# predicts from file provided by user
@app.post('/predict_file')
def prediction(file: UploadFile):
    model = app.state.model
    assert model is not None
    img = Image.open(file.file)
    #import ipdb; ipdb.set_trace()
    prediction = predict_image_model(img, model)
    return (prediction[0]).tolist()


#root endpoint
@app.get('/')
def root():
    return dict(greeting='Shark!')
