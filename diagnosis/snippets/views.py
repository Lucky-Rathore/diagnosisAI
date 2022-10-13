from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from snippets.models import Snippet
from snippets.serializers import SnippetSerializer

import numpy as np
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

import base64
import uuid

from threading import Lock, Thread
from typing import Optional

from django.conf import settings
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
        model = load_model(f'{settings.STATIC_ROOT}/model.h5', compile=False)
        model.compile(optimizer='adam', loss = self.get_weighted_loss(pos_weights, neg_weights))
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
    with open(f"{str(settings.STATIC_ROOT)}/{imageName}", "wb") as fh:
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

@csrf_exempt
def snippet_detail(request, pk):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        snippet = Snippet.objects.get(pk=pk)
    except Snippet.DoesNotExist:
        return HttpResponse(status=404)

    if request.method == 'GET':
        serializer = SnippetSerializer(snippet)
        return JsonResponse(serializer.data)

    elif request.method == 'PUT':
        data = JSONParser().parse(request)
        serializer = SnippetSerializer(snippet, data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data)
        return JsonResponse(serializer.errors, status=400)

    elif request.method == 'DELETE':
        snippet.delete()
        return HttpResponse(status=204)

@csrf_exempt
def snippet_list(request):
    if request.method == 'GET':
        b = compute_gradcam(Singleton(None).value, '/00000003_001.png', str(settings.STATIC_ROOT), getLabels(), selected_labels = [])
        return JsonResponse({"a": b}, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        #serializer = SnippetSerializer(data=data)
        img = data['img'].encode('utf-8');     
        imageName = f'{uuid.uuid4()}-{datetime.today()}.png'   
        storeImage(img, imageName)
        b = compute_gradcam(Singleton(None).value, imageName, str(settings.STATIC_ROOT), getLabels(), selected_labels = [])
        return JsonResponse({'predictions': b}, status=200)