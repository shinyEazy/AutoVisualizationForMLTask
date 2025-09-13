import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

API_ENDPOINT = "http://127.0.0.1:8000" 
API_INFO = """
{
  "predict_endpoint": {
    "path": "/predict",
    "method": "POST",
    "operationId": "MultimodalClassificationService__predict",
    "summary": "",
    "tags": [
      "Service APIs"
    ],
    "requestBody": {
      "content": {
        "multipart/form-data": {
          "schema": {
            "type": "object",
            "title": "Input",
            "required": [
              "image_zip",
              "labels_csv"
            ],
            "properties": {
              "image_zip": {
                "description": "A ZIP file containing all the images for multimodal classification task.",
                "format": "binary",
                "title": "Image Zip",
                "type": "string"
              },
              "labels_csv": {
                "description": "A csv file for multimodal classification task.",
                "format": "binary",
                "title": "Labels Csv",
                "type": "string"
              }
            }
          }
        }
      }
    },
    "responses": [
      "200",
      "400",
      "404",
      "500"
    ]
  }
}
"""
API_REQUEST_SAMPLE = """
curl -X 'POST' \
  'http://47.186.63.142:52901/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image_zip=@super_small_image.zip;type=application/x-zip-compressed' \
  -F 'labels_csv=@super_small_data.csv;type=text/csv'
"""
API_RESPONSE_SAMPLE = """
{
    "status": "success",
    "infer_time": 0.5484125036746264,
    "predictions": [
        {
            "key": 0,
            "class": 0,
            "confidence": 0.94629967212677
        },
        {
            "key": 1,
            "class": 1,
            "confidence": 0.6092917323112488
        }
    ]
}
"""

# Initialize the language model
llm = ChatOpenAI(
    model_name="gpt-5-nano",
    temperature=1
)

# Define a prompt template
prompt_template = PromptTemplate(
    input_variables=["input", "API_ENDPOINT", "API_INFO", "API_REQUEST_SAMPLE", "API_RESPONSE_SAMPLE"],
    template="""
Create a clean, modern, and intuitive HTML page designed to visualize API input and output data for machine learning tasks, emphasizing clarity and interactivity. The page should focus on presenting input data (e.g., multiple images, tabular data, or text) and corresponding output data (e.g., predictions, labels, or confidence scores) in a structured, easy-to-understand format. Ensure the layout is responsive, accessible, and visually appealing, using a card-based or row-based design to organize inputs and outputs clearly. Incorporate semantic colors, subtle animations, and interactive elements to enhance user experience while maintaining simplicity.

Use this information:  
API ENDPOINT: {API_ENDPOINT}  
API SCHEMA: {API_INFO}  
API REQUEST SAMPLE: {API_REQUEST_SAMPLE}  
API RESPONSE SAMPLE: {API_RESPONSE_SAMPLE}

The UI should include:  
1. **Header Section**:  
   - Display the API endpoint and a clear title (e.g., "API Data Visualizer").  
   - Include a status indicator (e.g., API connection status).  

2. **Input Visualization Section**:  
   - Support multiple input types (e.g., images, CSV, text) displayed in a grid or row-based layout.  
   - For images, show thumbnails in a responsive grid or rows with drag-and-drop upload support.  
   - For tabular data, display an interactive table with sorting and filtering.  
   - Provide real-time validation feedback (e.g., file type, size) with icons or color cues.  
   - Show a summary of input data (e.g., number of images, rows in CSV, or text length).  

3. **Output Visualization Section**:  
   - Display outputs (e.g., labels, predictions, confidence scores) directly aligned with corresponding inputs (e.g., labels below each image in a row).  
   - Use visual indicators like progress bars, badges, or color-coded labels for predictions (e.g., green for positive, red for negative).  
   - For JSON outputs, include an expandable/collapsible JSON viewer.  
   - Support interactive charts (e.g., bar charts for confidence scores) for numerical outputs.  

4. **Interactive Features**:  
   - A "Predict" button with loading states and progress indicators.  
   - Clickable inputs (e.g., image thumbnails) to trigger predictions or display details.  
   - Copy-to-clipboard functionality for API request/response data.  
   - Export options for output data (e.g., CSV, JSON).  
   - Collapsible sections for detailed inspection of inputs/outputs.  

5. **Modern Design Elements**:  
   - Use a clean, consistent typography with clear hierarchy (e.g., sans-serif fonts like Roboto).  
   - Apply a semantic color scheme (e.g., blue for inputs, green for outputs, red for errors).  
   - Use card-based or row-based layouts with subtle shadows, borders, and hover effects.  
   - Include icons (e.g., upload, success, error) for visual clarity.  
   - Add smooth transitions/animations for button clicks, loading states, and collapsible sections.  

6. **Responsive & Accessible Design**:  
   - Ensure a mobile-first layout that adapts to different screen sizes.  
   - Maintain high contrast ratios for readability and accessibility.  
   - Support keyboard navigation and screen readers with proper ARIA attributes.  
   - Use touch-friendly elements (e.g., larger buttons for mobile).  

7. **Data Visualization Components**:  
   - For multiple images, display each in a row or grid with corresponding output (e.g., predicted label) below or beside it.  
   - For tabular data, use interactive tables with pagination and sorting.  
   - For numerical outputs, include progress bars or simple charts (e.g., bar or pie charts).  
   - For classification tasks, use color-coded badges or icons to indicate results.  

8. **Error Handling & Feedback**:  
   - Show clear error messages with actionable suggestions (e.g., "Invalid file format, please upload PNG/JPEG").  
   - Use visual indicators for success (green checkmark) or error (red alert) states.  
   - Provide real-time feedback for invalid inputs or network issues.  

**Example Scenario**: For a machine learning task with multiple image inputs, display each image in a row with a thumbnail. Below each image, show a "Predict" button. After clicking, display the predicted label (e.g., "Cat: 95%") and confidence score (e.g., progress bar) directly below the corresponding image.

**Requirements**:  
- Return raw HTML code without markdown or code blocks.  
- Include inline CSS and JavaScript for a self-contained page.  
- Ensure the design is simple yet visually appealing, prioritizing data clarity and interactivity.  
- Handle dynamic inputs/outputs (e.g., variable number of images or rows) effectively.

User requirement: {input}
    """
)

def run_agent(user_prompt):
    prompt = prompt_template.format(input=user_prompt, API_ENDPOINT=API_ENDPOINT, API_INFO=API_INFO, API_REQUEST_SAMPLE=API_REQUEST_SAMPLE, API_RESPONSE_SAMPLE=API_RESPONSE_SAMPLE)
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    user_prompt = "Generate a comprehensive HTML UI for the multimodal classification API endpoint"
    try:
        html_content = run_agent(user_prompt)
        with open("index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully wrote response to index.html")
    except Exception as e:
        print(f"Error writing to index.html: {str(e)}")