import torch
import os
import logging
import base64
import io
import json
import utils
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sagemaker.serializers import NumpySerializer
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
JPEG_CONTENT_TYPE = 'application/x-image'
logger = logging.getLogger()
 
def model_fn(model_dir):
    if 'COMPILEDMODEL' in os.environ and os.environ['COMPILEDMODEL'] == 'True':
        import neopytorch
        if 'INFERENTIA' in os.environ and os.environ['INFERENTIA'] == 'True':
            ssd_model = torch.jit.load('model_neuron.pt', map_location=device)
        else:
            neopytorch.config(model_dir=model_dir, neo_runtime=True)
            ssd_model = torch.jit.load(os.path.join(model_dir, 'compiled.pt'), map_location=device)
        return ssd_model.to(device)
 
    # Model
    model = torch.jit.load('model.pth', map_location=device)
    model.to(device)
    return model 
 
def predict_fn(image, model):
    return model.forward(image)
 
def input_fn(request_body, content_type):
    if content_type == JPEG_CONTENT_TYPE:
        iobytes = io.BytesIO(request_body)
        decoded = Image.open(iobytes)
        preprocess = transforms.Compose([
            transforms.Resize(416),
            transforms.CenterCrop(416),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[ 0.229, 0.224, 0.225]
            )
        ])
        normalized = preprocess(decoded)
        batchified = normalized.unsqueeze(0)
        return batchified.to(device)
 
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def output_fn(data, accept):
    list_features_numpy = []
    for feature in data:
        list_features_numpy.append(feature.data.cpu().numpy())

    img = np.zeros(shape=(3, 416, 416)) #Image size = 416 x 416
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    resp = utils.post_processing(img, 0.3, 80, 0.4, list_features_numpy)
    logger.info(resp)
    ser = NumpySerializer(content_type='application/x-npy', dtype=np.ndarray)
    return ser.serialize(data=np.array(resp))