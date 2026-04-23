from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import requests
from bs4 import BeautifulSoup
import datetime
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

# Your original working code goes here...
# Copy and paste ALL your original code from the main.py that was working
# This will be one large file but it will work immediately