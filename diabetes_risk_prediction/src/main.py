import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

if __name__ == '__main__':
    # Launch Gradio interface from notebooks/app.py
    notebooks_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks')
    sys.path.append(notebooks_dir)
    from app import demo
    print("Starting Diabetes Prediction Gradio Web App...")
    demo.launch(inbrowser=True)
