import gradio as gr
import numpy as np
from PIL import Image

def mock_predict(image):
    classes = ["Cat", "Dog", "Bird"]
    return np.random.choice(classes)

def classify_image(image):
    if image is None:
        return "No image uploaded. Please upload an image."
    
    try:
        image = Image.fromarray(image)
        prediction = mock_predict(image)
        return prediction
    except Exception as e:
        return f"Error processing the image: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Image Classifier")
    gr.Markdown("Upload an image to classify it as a Cat, Dog, or Bird.")
    
    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Image")
        classify_button = gr.Button("Classify")
    
    output_label = gr.Label(label="Prediction")
    
    classify_button.click(fn=classify_image, inputs=image_input, outputs=output_label)

demo.launch()