import os
import traceback

from io import BytesIO
from PIL import Image
from typing import Union, Literal, List

from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.openapi.docs import get_swagger_ui_html

import uvicorn
import pandas as pd

from src.modeling.search import Search
from src.api.templates import OutputAPI
from src.utils.io import load_config, load_multipart_file
from src.utils.connection import download_folder_from_s3
from src.utils.converters import image2base64


API_CONFIG = load_config(path='config/api_config.yaml')
API_RESPONDER = OutputAPI()


app = FastAPI(
      root_path =  os.getenv("ROOT_PATH"), 
          title = API_CONFIG['DESCRIPTION']['title'],
    description = API_CONFIG['DESCRIPTION']['overview'],
   openapi_tags = API_CONFIG['TAGS'],
        version = "1.0.0", 
       docs_url = None, 
      redoc_url = None,
)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    query_params = str(request.query_params)
    openapi_url = app.root_path + app.openapi_url + "?" + query_params
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="DataSpire - Documents",
        swagger_favicon_url = "https://assets.dataspire.ai/static/favicon.ico",
    )
    
    
@app.post("/search/text")
async def search_by_text(    
    content:                        str    = Form(description=API_CONFIG['PARAMETERS']['content']),
      top_k:                        int    = Form(description=API_CONFIG['PARAMETERS']['top_k'], default=5),
    storage: Literal['local', 'dataspire'] = Form(description=API_CONFIG['PARAMETERS']['storage'], default='local'),
):
    try:

        # Check params
        storage = storage.lower()
        if storage != "local":
            raise ValueError(f'storage is not supported with {storage}')

        if type(content) != str:
            raise TypeError('text must be string')

        # Run ranking
        result_list = MODEL.rank_results(content, n=top_k)

        # Format output
        results = [
            {
                'index': ri+1,
                'path': r.image_path,
                'score': r.score,
                'binary': str(image2base64(Image.open('./' + r.image_path.replace('\\', '/'))))[2:-1],
            } for ri, r in enumerate(result_list)
        ]
        response = API_RESPONDER.result(is_successful=True, data=results)

    except Exception as e:
        print(e)
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())

    return response

    
@app.post("/search/image")
async def search_by_image(    
    content:                    UploadFile = File(description=API_CONFIG['PARAMETERS']['content'], media_type='multipart/form-data'),
      top_k:                        int    = Form(description=API_CONFIG['PARAMETERS']['top_k'], default=5),
    storage: Literal['local', 'dataspire'] = Form(description=API_CONFIG['PARAMETERS']['storage'], default='local'),
):
    try:

        # Check params
        storage = storage.lower()
        if storage != "local":
            raise ValueError(f'storage is not supported with {storage}')
        
        # Preprocess
        filename = content.filename
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in [".png", ".jpg", ".jpeg"]:
            raise TypeError(f"{filename} with type")

        content = await content.read()
        content = Image.open(BytesIO(content)).convert('RGB')

        # Run ranking
        result_list = MODEL.rank_results(content, n=top_k)

        # Format output
        results = [
            {
                'index': ri+1,
                'path': r.image_path,
                'score': r.score,
                'binary': str(image2base64(Image.open('./' + r.image_path.replace('\\', '/'))))[2:-1],
            } for ri, r in enumerate(result_list)
        ]
        response = API_RESPONDER.result(is_successful=True, data=results)

    except Exception as e:
        print(e)
        response = API_RESPONDER.result(is_successful=False, err_log=traceback.format_exc())

    return response


if __name__ == "__main__":
    
    # Read AWS credentials
    # BUCKET_NAME = os.getenv("ARTIFACT_BUCKET")
    # MODEL_NAME = os.getenv("MODEL_NAME")
    # MODEL_VERSION = os.getenv("ARTIFACT_VERSION")

    # AWSS3_PATH =  MODEL_NAME + "/" + MODEL_VERSION + "/"
    # AWSS3_CREDENTIAL = {
    #     'BUCKET_NAME': BUCKET_NAME,
    # }

    # Download artefact
    LOCAL_PATH = 'artefacts/'
    # if os.path.isdir(LOCAL_PATH) is False:
    #     os.makedirs(LOCAL_PATH)
    # download_folder_from_s3(LOCAL_PATH, AWSS3_PATH, AWSS3_CREDENTIAL)

    # Load artefacts
    import yaml
    with open('config/model_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    MODEL = Search(image_dir_path=config['database']['path'], 
                         traverse=config['database']['traverse'],
                       model_path=config['multimodal_embedding'],)

    # Run application
    uvicorn.run(app, **API_CONFIG['SERVER'])


