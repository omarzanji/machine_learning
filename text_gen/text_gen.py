import random
import json
import numpy as np
import tensorflow as tf


with open("./database/news_0000008.json", 'r') as f:
    data = json.load(f)

text = data['text']
