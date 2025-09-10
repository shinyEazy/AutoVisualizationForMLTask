import gradio as gr
import requests

API_URL = "http://136.60.217.200:50318/predict"

def predict(image_path):
    try:
        with open(image_path, "rb") as image_file:
            files = {'images': image_file}
            response = requests.post(API_URL, files=files, timeout=15)
            response.raise_for_status()  # Raise an error for bad responses
            try:
                result = response.json()
            except ValueError:
                return f"Error parsing response: {response.text}"
            
            if result.get("status") == "success":
                predictions = result.get("predictions", [])
                if predictions:
                    return f"Predicted class: {predictions[0]['class']}, Confidence: {predictions[0]['confidence']:.2f}"
                else:
                    return "No predictions found."
            else:
                return "Prediction failed. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error during API request: {str(e)}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Image Classifier")
    gr.Markdown("Upload an image to classify it.")
    
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Image")
        classify_button = gr.Button("Classify")
    
    output_text = gr.Textbox(label="Prediction Result", interactive=False)
    
    classify_button.click(fn=predict, inputs=image_input, outputs=output_text)

demo.launch()