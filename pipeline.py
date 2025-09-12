import argparse
import dspy
from dotenv import load_dotenv
import os

load_dotenv()

lm = dspy.LM(model="gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)

class TaskAnalysis(dspy.Signature):
    """Analyze the ML task and input type to determine required Gradio components."""
    task_name = dspy.InputField(desc="Name of the ML task")
    input_type = dspy.InputField(desc="Type of input data (e.g., image, text, tabular)")
    components = dspy.OutputField(desc="List of Gradio components needed for this task")

class GradioCodeGenerator(dspy.Signature):
    """Generate complete Gradio code based on task requirements."""
    task_name = dspy.InputField(desc="Name of the ML task")
    input_type = dspy.InputField(desc="Type of input data")
    components = dspy.InputField(desc="List of required Gradio components")
    api_endpoint = dspy.InputField(desc="API endpoint for model inference")
    gradio_code = dspy.OutputField(desc="Complete Python code for Gradio app")

def analyze_task(task_name, input_type):
    """Analyze the task and determine required components."""
    analysis = dspy.Predict(TaskAnalysis)(
        task_name=task_name,
        input_type=input_type
    )
    return analysis

def generate_gradio_code(task_name, input_type, components, api_endpoint):
    """Generate Gradio code based on task analysis."""
    code_generation = dspy.Predict(GradioCodeGenerator)(
        task_name=task_name,
        input_type=input_type,
        components=components,
        api_endpoint=api_endpoint
    )
    return code_generation.gradio_code

def main():
    parser = argparse.ArgumentParser(
        description="CLI for generating and running Gradio apps from ML task descriptions"
    )
    parser.add_argument("--task_name", required=True, help="ML task description")
    parser.add_argument("--input_type", required=True, help='Input type e.g. "image", "text", "tabular"')
    parser.add_argument("--api_endpoint", required=True, help='API endpoint for model inference')

    args = parser.parse_args()

    try:
        # Step 1: Analyze the task
        print(f"Analyzing task: {args.task_name} with input type: {args.input_type}")
        analysis = analyze_task(args.task_name, args.input_type)
        
        print(f"Recommended components: {analysis.components}")
        
        # Step 2: Generate Gradio code
        print("Generating Gradio code...")
        gradio_code = generate_gradio_code(
            args.task_name, 
            args.input_type, 
            analysis.components,
            args.api_endpoint
        )
        
        # Step 3: Save the generated code
        output_file = "app.py"
        with open(output_file, "w") as f:
            f.write(gradio_code)
            
        print(f"Gradio app generated successfully! Saved to {output_file}")
    except Exception as e:
        print(f"Error processing task: {str(e)}")

if __name__ == "__main__":
    main()

# python .\main.py --task_name "MULTIMODAL_CLASSIFICATION22" --input_type "image, text" --api_endpoint "127.0.0.1/predict"