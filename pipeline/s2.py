import os
import time
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
    temperature=1,
)

prompt_template = PromptTemplate(
    input_variables=["input", "API_ENDPOINT", "API_INFO", "API_REQUEST_SAMPLE", "API_RESPONSE_SAMPLE"],
    template="""
Generate a clean, modern HTML interface focused solely on data visualization for ML API inputs and outputs, with a strict vertical layout and no horizontal spreading.

## API Context
- Endpoint: {API_ENDPOINT}
- Schema: {API_INFO}
- Request Sample: {API_REQUEST_SAMPLE}
- Response Sample: {API_RESPONSE_SAMPLE}

## Core Data Visualization Focus
### 1. Input Data Rendering
Based on the API schema, identify the input type and render it appropriately:
- ZIP Files: Implement a drag-and-drop upload zone that displays the file count and total size after upload.
- Tabular Data: Include a file upload with an immediate table preview in a scrollable container with fixed height and vertical scrolling. (if cell is to long, use truncate for )
- Image Data: Use a drag-and-drop upload with image previews in a carousel format for multiple images, scaling images to fit neatly.

### 2. Output Data Rendering
Based on the response sample, display output data in the most suitable format:
- If input contain tabular and response is label for each row, then combine those into a table and display it.

### 3. Simple Interactive Features
Include only essential interactions derived from the API schema:
- Predict Button: A button to trigger the API call.
- Clear Button: Reset all inputs and outputs.
- Download Results: Allow downloading of output data.
- Copy Data: Provide buttons to copy JSON output.

### 4. Clean Design System
- Layout: Use a strict single-column layout with `display: flex; flex-direction: column;`. Stack all sections vertically with no horizontal splits. Each section should be inside a card with consistent padding and spacing.
- Colors: 
  - Blue for input sections and interactive actions.
  - Green for output sections and success states.
  - Red for errors and warnings.
  - Gray for neutral text and borders.
- Typography: Use font mono-space with clear size hierarchy and high contrast for readability.
- Responsiveness: Ensure the interface works on mobile and desktop, with vertical scrolling for any overflow content.

### 5. Technical Requirements
- HTML5: Use semantic elements (e.g., <section>, <article>) for better structure.
- Inline Styles: Implement all CSS inline to keep the HTML self-contained; no external libraries or stylesheets, use tailwind CSS only, don't use global CSS.
- JavaScript: Use minimal inline JavaScript for interactivity, such as handling file uploads and API calls. Avoid complex frameworks.
- Performance: Optimize for fast loading and efficient rendering of large datasets without lag.
- Make sure: `max-width: 1240px;` is set in the root element.

## User Requirements
- Specific Request: {input}

## Implementation Focus
- Data Visualization Only: Focus on rendering input and output data clearly without any charts, graphs, or statistical analysis.
- Simple Interface: Keep the design minimal and functional.
- API Schema Based: Create only input components specified in the API schema; avoid unnecessary elements.
- Vertical Layout Everywhere: Enforce a vertical layout in all sections using CSS flexbox or similar for vertical flow. Use vertical scrolling for any content overflows; avoid horizontal scrolling.

## Example for different input types
1. Tabular
```html

```

2. ZIP
```html

```
"""
)

def run_agent(user_prompt):
    start_time = time.time()
    prompt = prompt_template.format(
        input=user_prompt,
        API_ENDPOINT=API_ENDPOINT,
        API_INFO=API_INFO,
        API_REQUEST_SAMPLE=API_REQUEST_SAMPLE,
        API_RESPONSE_SAMPLE=API_RESPONSE_SAMPLE
    )
    response = llm.invoke(prompt)
    end_time = time.time()
    duration = end_time - start_time
    return response.content, duration


if __name__ == "__main__":
    user_prompt = "Generate a comprehensive HTML UI for multimodal classification task."
    try:
        html_content, elapsed = run_agent(user_prompt)
        with open("index.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Successfully wrote to index.html")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error writing to index.html: {str(e)}")