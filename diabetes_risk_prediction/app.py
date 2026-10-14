import importlib.util
import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
target_app = os.path.join(base_dir, 'notebooks', 'app.py')

spec = importlib.util.spec_from_file_location("diabetes_gradio_app", target_app)
diabetes_gradio_app = importlib.util.module_from_spec(spec)
sys.modules["diabetes_gradio_app"] = diabetes_gradio_app
spec.loader.exec_module(diabetes_gradio_app)

if __name__ == '__main__':
    print("Launching Gradio Diabetes Risk Prediction App...")
    diabetes_gradio_app.demo.launch(inbrowser=True)
