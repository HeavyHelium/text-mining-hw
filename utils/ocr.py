import requests
import os, sys
import pandas as pd
import re, json
import pytesseract
import matplotlib.pyplot as plt


from dotenv import load_dotenv
from PIL import Image
from tempfile import NamedTemporaryFile
from io import BytesIO


def ocr_space_file(filename: str, api_key: str,
                   language: str = 'eng', overlay: bool = False) -> dict:
    """
    Sends a local image file to OCR.space API and returns the JSON response.
    """
    url = 'https://api.ocr.space/parse/image'
    with open(filename, 'rb') as f:
        payload = {
            'apikey': api_key,
            'language': language,
            'isOverlayRequired': overlay
        }
        files = {
            'file': (filename, f)
        }
        response = requests.post(url, data=payload, files=files)
    response.raise_for_status() 
    return response.json()


def ocr_df(df, lang='bul'):
    ocr_results = []
    for _, row in df.iterrows():
        image = Image.open(BytesIO(row['image']['bytes']))
        text = pytesseract.image_to_string(image, lang=lang)
        ocr_results.append(text)
    return ocr_results
