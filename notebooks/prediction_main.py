import numpy as np
import pandas as pd
import pyarrow as pa

from sherlock import helpers
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings

###########################################################################################
import json
from collections import MutableMapping
from contextlib import suppress

from enum import Enum
import shutil
import os
from fastapi import FastAPI, File, Request
from pydantic import BaseModel
###########################################################################################







##########################################################################################

#def create_json(json_str):
#    json_dct =  json.loads(json_str)
#    return json_dct
##########################################################################################



####################################################
# downloads embedding models, 1st time only
#####################################################
prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()

######################################################



def do_pred(req_info):
    
    df = pd.Series(req_info , name="values")

    # extract features

    extract_features(
        "../temporary.csv",
        df
    )
    feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)


    ####################################################################
    model = SherlockModel()
    model.initialize_model_from_json(with_weights=True, model_id="sherlock")

    predicted_labels = model.predict(feature_vectors, "sherlock")
    return predicted_labels

########################################################################

#class test_data(BaseModel):
#    info: list


app = FastAPI()


@app.post("/get_pred")


async def get_pred(info : list):

    req_info = info # .json()
    #return req_info
    
    pred_op = do_pred(req_info)
    

    return {"predicted_columns": list(pred_op)}



@app.get("/getInformation")
async def getInformation():
    return {"columns_trained" : ['address', 'affiliate', 'affiliation', 'age', 'album', 'area',
       'artist', 'birth date', 'birth place', 'brand', 'capacity',
       'category', 'city', 'class', 'classification', 'club', 'code',
       'collection', 'command', 'company', 'component', 'continent',
       'country', 'county', 'creator', 'credit', 'currency', 'day',
       'depth', 'description', 'director', 'duration', 'education',
       'elevation', 'family', 'file size', 'format', 'gender', 'genre',
       'grades', 'industry', 'isbn', 'jockey', 'language', 'location',
       'manufacturer', 'name', 'nationality', 'notes', 'operator',
       'order', 'organisation', 'origin', 'owner', 'person', 'plays',
       'position', 'product', 'publisher', 'range', 'rank', 'ranking',
       'region', 'religion', 'requirement', 'result', 'sales', 'service',
       'sex', 'species', 'state', 'status', 'symbol', 'team', 'team name',
       'type', 'weight', 'year']}