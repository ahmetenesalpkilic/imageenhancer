# sr_dataset.py dosyası

import torch
from torch.utils.data import Dataset # PyTorch'ta veri yönetimi için ana sınıf
import cv2
import glob
import os
import random
import numpy as np