
import numpy as np
import pandas as pd
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

import base64
import uuid

from threading import Lock, Thread
from typing import Optional
from datetime import datetime
from base64 import decodestring



class SingletonMeta(type):
    _instance = None
    _lock: Lock = Lock()
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__call__(*args, **kwargs)
        return cls._instance

class Singleton(metaclass=SingletonMeta):
    value: object = None
    """
    We'll use this property to prove that our Singleton really works.
    """
    
    def __init__(self, value: object) -> None:
        model = DiagnosisModel()
        self.value = model.getModel()

    def some_business_logic(self):
        """
        Finally, any singleton should define some business logic, which can be
        executed on its instance.
        """

class DiagnosisModel():
    def get_weighted_loss(self, pos_weights, neg_weights, epsilon=1e-7):
        def weighted_loss(y_true, y_pred):
            # initialize loss to zero
            loss = 0.0
            logY = K.log(y_pred + epsilon)         
            logYN = K.log(1- y_pred  + epsilon)      
            
            for i in range(len(pos_weights)):
                loss += K.mean(
                    -1 * (
                pos_weights[i] * y_true[:,i] * logY[:,i] + neg_weights[i] * (1 - y_true[:,i]) * logYN[:,i]
                )            
                ) 
            return loss
        return weighted_loss
    
    def getLabels(self):
        labels = ['Cardiomegaly', 
            'Emphysema', 
            'Effusion', 
            'Hernia', 
            'Infiltration', 
            'Mass', 
            'Nodule', 
            'Atelectasis',
            'Pneumothorax',
            'Pleural_Thickening', 
            'Pneumonia', 
            'Fibrosis', 
            'Edema', 
            'Consolidation']
        return labels

    def getModel(self):        
        neg_weights = [0.11904762, 0.13333333, 0.12619048, 0.11904762, 0.14047619, 0.14285714, 0.12857143, 0.14285714, 0.13095238, 0.13809524, 0.11904762, 0.1452381 , 0.11904762, 0.12619048]
        pos_weights = [0.88095238, 0.86666667, 0.87380952, 0.88095238, 0.85952381, 0.85714286, 0.87142857, 0.85714286, 0.86904762, 0.86190476, 0.88095238, 0.8547619 , 0.88095238, 0.87380952]
        base_model = DenseNet121(weights=f'../staticfiles/densenet.hdf5', include_top=False)
        poolLayer = GlobalAveragePooling2D()(base_model.output) # add a global spatial average pooling layer
        predictions = Dense(len(self.getLabels()), activation="sigmoid")(poolLayer) # and a logistic layer
        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer='adam', loss = self.get_weighted_loss(pos_weights, neg_weights))
        model.load_weights(f"../staticfiles/pretrained_model.h5")
        return model  

def load_image(img, image_dir,preprocess=True, H=320, W=320):
        """Load and preprocess image."""
        img_path = image_dir +"/" + img
        mean = np.array(122.40529296875)
        std = np.array(70.33326676182992)
        x = image.load_img(img_path, target_size=(H, W))
        if preprocess:
            x -= mean
            x /= std
            x = np.expand_dims(x, axis=0)
        return x

def compute_gradcam(model, img, image_dir, labels, selected_labels, layer_name='bn'):
    preprocessed_input = load_image(img, image_dir)
    predictions = model.predict(preprocessed_input)
    return str(predictions)

def storeImage(img, imageName):
    print("log: storing image")
    with open(f"../staticfiles/{imageName}", "wb") as fh:
        fh.write(decodestring(img))

def getLabels():
        labels = ['Cardiomegaly', 
            'Emphysema', 
            'Effusion', 
            'Hernia', 
            'Infiltration', 
            'Mass', 
            'Nodule', 
            'Atelectasis',
            'Pneumothorax',
            'Pleural_Thickening', 
            'Pneumonia', 
            'Fibrosis', 
            'Edema', 
            'Consolidation']
        return labels

