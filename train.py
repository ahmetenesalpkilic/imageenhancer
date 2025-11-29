# train.py dosyası

from sr_dataset import SRDataset # Kendi veri setimiz
from srcnn_model import SRCNN     # Kendi modelimiz
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os