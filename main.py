import os
import re
import subprocess
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import warnings
from dotenv import load_dotenv
import sys
import argparse
import signal
import json

warnings.filterwarnings("ignore")

load_dotenv()

# Agent 1: Requirements Analysis Agent
class RequirementsAnalysisAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.prompt_template = PromptTemplate(
            input_variables=["task"],
            template="""
            You are an expert at analyzing machine learning tasks and determining what Gradio interface components are needed.
            
            Based on the following ML task description: {task}
            
            Please analyze and determine:
            1. What input components are needed (e.g., upload button, textbox, microphone, etc.)
            2. What output components are needed (e.g., label, image, audio, plot, etc.)
            3. Any special processing requirements
            4. What the mock model function should return for demonstration purposes
            
            Format your response clearly with sections for Inputs, Outputs, Processing, and MockFunction.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def analyze(self, task):
        return self.chain.run(task=task)

# Agent 2: Code Generation Agent
class CodeGenerationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.prompt_template = PromptTemplate(
            input_variables=["analysis", "error_log", "api_info"],
            template="""
            You are an expert at generating Gradio code based on requirements analysis.
            
            Previous analysis: {analysis}
            
            {error_log}
            
            External prediction API information (if provided):
            {api_info}
            
            Generate a complete, runnable Gradio app in a single Python file. Follow these guidelines:
            1. Use the latest gradio library with Blocks API (not Interface)
            2. Use gr.Blocks() as the main container
            3. Include a mock model function that simulates the ML task
            4. Make the UI clean and user-friendly with proper layout components
            5. Handle edge cases and provide helpful error messages
            6. Include necessary imports
            7. The code should be executable directly
            8. Use the following structure:
               with gr.Blocks() as demo:
                   # Add components here
                   ...
               demo.launch()
            9. If API information is provided above, DO NOT mock. Implement a predict() that calls the HTTP API using requests with multipart/form-data. Use the field name images and send the uploaded file as raw bytes. Set the input image component to gr.Image(type="filepath") so you receive a filesystem path. Open the file in binary mode and build files with key images containing the file data. Use requests.post(url, files=files, timeout=15) and close the file handle after. Try response.json() first; if it fails, fall back to response.text and display it. Include graceful error handling and timeouts. If the API probe failed due to missing test files, still implement the API call correctly - the user will provide their own images.
            
            Return only the Python code without any explanations or markdown formatting.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def generate_code(self, analysis, error_log="", api_info=""):
        if error_log:
            error_log = f"Previous execution errors that need to be fixed:\n{error_log}"
        
        response = self.chain.run(analysis=analysis, error_log=error_log, api_info=api_info)
        # Extract code from response (in case the model adds explanations)
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        else:
            return response

# Gradio App Manager
class GradioAppManager:
    def __init__(self):
        self.current_process = None
        self.error_log = ""
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.pid_file_path = os.path.join(self.project_root, '.gradio_app.pid')
    
    def run_gradio_code(self, code):
        # Write code directly to app.py in the current project directory
        app_file_path = os.path.join(self.project_root, 'app.py')
        try:
            with open(app_file_path, 'w', encoding='utf-8') as f:
                f.write(code)
        except Exception as e:
            error_msg = f"Error writing to app.py: {str(e)}"
            self.error_log = error_msg
            return error_msg
        
        try:
            # Terminate previous process if exists
            if self.current_process:
                self.current_process.terminate()
            
            # Run the Gradio app in a subprocess
            self.current_process = subprocess.Popen(
                [sys.executable, app_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Persist PID so we can stop later via CLI
            try:
                with open(self.pid_file_path, 'w', encoding='utf-8') as pidf:
                    pidf.write(str(self.current_process.pid))
            except Exception:
                pass
            
            # Wait a bit to capture any immediate errors
            try:
                stdout, stderr = self.current_process.communicate(timeout=10)
                if stderr:
                    self.error_log = stderr
                    return f"Error starting app: {stderr}"
                else:
                    self.error_log = ""
                    return "App started successfully!"
            except subprocess.TimeoutExpired:
                # If no immediate errors, the app is probably running
                self.error_log = ""
                return "App is running in the background!"
                
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            self.error_log = error_msg
            return error_msg
    
    def stop_gradio_app(self):
        # Prefer in-memory process if available
        if self.current_process:
            try:
                self.current_process.terminate()
            except Exception:
                pass
            self.current_process = None
            try:
                if os.path.exists(self.pid_file_path):
                    os.remove(self.pid_file_path)
            except Exception:
                pass
            return "Gradio app stopped!"
        # Fallback to PID file termination
        if os.path.exists(self.pid_file_path):
            try:
                with open(self.pid_file_path, 'r', encoding='utf-8') as pidf:
                    pid_str = pidf.read().strip()
                if pid_str:
                    pid = int(pid_str)
                    try:
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        else:
                            os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
                os.remove(self.pid_file_path)
                return "Gradio app stopped!"
            except Exception as e:
                return f"Failed to stop app: {str(e)}"
        return "No Gradio app running!"

# Agent 3: External API Probe Agent
class ExternalAPIProbeAgent:
    def __init__(self, default_url: str = "http://136.60.217.200:50318/predict", default_image_path: str = r"C:/Users/ASUS/Downloads/images.webp"):
        self.default_url = default_url
        self.default_image_path = default_image_path

    def probe(self, url: str | None = None, image_path: str | None = None) -> str:
        target_url = url or self.default_url
        path = image_path or self.default_image_path
        
        # Check if image file exists
        if not os.path.exists(path):
            return json.dumps({
                "url": target_url, 
                "image_path": path, 
                "error": f"Image file not found: {path}",
                "note": "API expects multipart form-data with field 'images' containing image file bytes"
            })
        
        # Build curl command - use curl.exe on Windows to avoid PowerShell alias
        # Convert Windows path to forward slashes for curl
        curl_path = path.replace('\\', '/')
        if os.name == 'nt':
            cmd = [
                "curl.exe",
                "-X", "POST", target_url,
                "-F", f"images=@{curl_path}"
            ]
        else:
            cmd = [
                "curl",
                "-X", "POST", target_url,
                "-F", f"images=@{curl_path}"
            ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            info = {"url": target_url, "image_path": path, "curl_path": curl_path, "stdout": stdout, "stderr": stderr, "returncode": result.returncode}
            # Try parse JSON if possible
            try:
                if stdout:
                    info["json"] = json.loads(stdout)
            except Exception:
                pass
            return json.dumps(info, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"url": target_url, "image_path": path, "curl_path": curl_path, "error": str(e)})

# Custom Prompt Template for the Orchestrator Agent
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        history = "\n".join([f"Agent: {action.log}\nObservation: {observation}" 
                            for action, observation in intermediate_steps])
        kwargs["history"] = history
        return self.template.format(**kwargs)

# Custom Output Parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if text.strip().startswith("Final Answer:"):
            return AgentFinish(return_values={"output": text.split("Final Answer:")[-1].strip()}, log=text)
        else:
            return AgentAction(tool="Gradio Code Generator", tool_input=text.strip(), log=text)

# Main Orchestrator Agent
class OrchestratorAgent:
    def __init__(self):
        self.requirements_agent = RequirementsAnalysisAgent()
        self.code_agent = CodeGenerationAgent()
        self.app_manager = GradioAppManager()
        self.api_probe_agent = ExternalAPIProbeAgent()
        
        tools = [
            Tool(
                name="Requirements Analyzer",
                func=self.requirements_agent.analyze,
                description="Analyzes ML tasks and determines Gradio components needed"
            ),
            Tool(
                name="Gradio Code Generator",
                func=self.generate_and_test_code,
                description="Generates and tests Gradio code based on requirements analysis"
            )
        ]
        
        tool_names = [tool.name for tool in tools]
        
        prompt_template = """
        You are orchestrating a multi-agent system to create Gradio applications for ML tasks.
        
        Your goal is to help users create Gradio interfaces for their machine learning tasks.
        
        You have access to the following tools:
        - Requirements Analyzer: Analyzes ML tasks and determines what Gradio components are needed
        - Gradio Code Generator: Generates and tests Gradio code based on requirements analysis
        
        Previous interactions:
        {history}
        
        User's task: {input}
        
        You should first use the Requirements Analyzer to understand what's needed, then use the Gradio Code Generator to create and test the application.
        
        If the Gradio Code Generator returns errors, you should analyze them and try again until the application runs successfully.
        
        When you have a successful application, provide the Final Answer with a summary of what was created.
        
        Response:
        """
        
        prompt = CustomPromptTemplate(
            template=prompt_template,
            input_variables=["input", "history", "intermediate_steps"]
        )
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.agent = LLMSingleActionAgent(
            llm_chain=LLMChain(llm=llm, prompt=prompt),
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            verbose=True,
            memory=ConversationBufferWindowMemory(k=3)
        )
    
    def generate_and_test_code(self, analysis):
        max_attempts = 3
        # Probe external API once to gather response shape
        try:
            api_probe_info = self.api_probe_agent.probe()
        except Exception as e:
            api_probe_info = f"API probe failed: {str(e)}"
        
        for attempt in range(max_attempts):
            code = self.code_agent.generate_code(analysis, self.app_manager.error_log, api_probe_info)
            result = self.app_manager.run_gradio_code(code)
            
            if "Error" not in result and "error" not in result.lower():
                return f"Code generated successfully! App status: {result}\n\nGenerated code:\n```python\n{code}\n```"
            elif attempt < max_attempts - 1:
                # Try again with the error log
                continue
            else:
                return f"Failed to generate working code after {max_attempts} attempts. Last error: {result}"
    
    def process_task(self, task):
        return self.agent_executor.run(input=task)

def main():
    parser = argparse.ArgumentParser(description="CLI for generating and running Gradio apps from ML task descriptions")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate", help="Analyze task, generate Gradio app code to app.py, and run it")
    gen_parser.add_argument("--task", required=True, help="ML task description")

    stop_parser = subparsers.add_parser("stop", help="Stop the currently running Gradio app")

    args = parser.parse_args()

    orchestrator = OrchestratorAgent()

    if args.command == "stop":
        result = orchestrator.app_manager.stop_gradio_app()
        print(result)
        return

    # Default to generate if no command provided, for convenience
    task = getattr(args, 'task', None)
    if args.command is None and not task:
        parser.error("--task is required when no command is specified")

    try:
        output = orchestrator.process_task(task)
        print(output)
    except Exception as e:
        print(f"Error processing task: {str(e)}")


if __name__ == "__main__":
    main()
