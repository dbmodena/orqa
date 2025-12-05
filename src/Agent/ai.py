import os 
from litellm import completion
import importlib
from pydantic import BaseModel
from typing import Optional, Dict, Any, Type
from pathlib import Path

import yaml
import json
import time



class LLMClient:
    """
    LiteLLM client with YAML configuration and structured output support.
    """
    
    def __init__(self, config_path: Path):
        """
        Initialize LLM client with configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path.name)
        self.model = self.config.get("model", "groq/llama-3.1-8b-instant")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        self.enable_json_mode = self.config.get("enable_json_mode", True)
        #self.prompt_key = self.config.get("prompt","You are an helpful assistant")
        self.response_model = self._load_response_model()
        if self.response_model is not None:
            self.raw=False
        else:
            self.raw=True
        api_keys = self.config.get("api_keys", {})
        if api_keys:
            for key_name, key_value in api_keys.items():
                if key_value:  # Only set if value is not None or empty
                    os.environ[key_name] = key_value
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _clean_json_response(self, content: str) -> str:
        """Clean up JSON response by removing markdown code blocks"""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()

    def _load_response_model(self) -> Optional[Type[BaseModel]]:
            """
            Dynamically load Pydantic model from config.
            
            Returns:
                Pydantic model class or None if not specified
            """
            response_model_config = self.config.get("response_model")
            
            if not response_model_config:
                return None
            
            module_name = response_model_config.get("module")
            class_name = response_model_config.get("class")
            
            if not module_name or not class_name:
                raise ValueError(
                    "response_model must specify both 'module' and 'class'"
                )
            
            try:
                # Import the module
                module = importlib.import_module(module_name)
                
                # Get the class from the module
                model_class = getattr(module, class_name)
                
                # Verify it's a Pydantic model
                if not issubclass(model_class, BaseModel):
                    raise TypeError(
                        f"{class_name} is not a Pydantic BaseModel"
                    )
                
                return model_class
                
            except ImportError as e:
                raise ImportError(
                    f"Could not import module '{module_name}': {e}"
                )
            except AttributeError as e:
                raise AttributeError(
                    f"Could not find class '{class_name}' in module '{module_name}': {e}"
                )


    def complete_prompt(self):
        print("hello")

    def complete(
        self, prompt:str,
        response_model: Optional[Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Make a completion request with optional structured output.
        
        Args:
            prompt: The prompt to send to the model
            response_model: Optional Pydantic model for structured output
            temperature: Override default temperature
            max_retries: Override default max_retries
            **kwargs: Additional arguments to pass to litellm.completion
        
        Returns:
            If response_model is provided: instance of the Pydantic model
            Otherwise: raw string response
        """
        temp = temperature if temperature is not None else self.temperature
        retries = max_retries if max_retries is not None else self.max_retries
        #prompt = create_analysis_prompt(csv_info,{json.dumps(schema, indent=2)})
        # Prepare completion arguments
        #messages = [
        #    {"role": "system", "content": system_prompt},
        #    {"role": "user", "content": user_prompt}
        #]
        completion_args = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            **kwargs
        }
        # Enable JSON mode if we have a response model
        if response_model and self.enable_json_mode:
            completion_args["response_format"] = {"type": "json_object"}
            self.response_model = response_model
            self.raw = False
        elif self.response_model:
            completion_args["response_format"] = self.response_model
        
        # Retry loop
        last_error = None
        for attempt in range(retries):
            try:
                print(f"Attempt {attempt + 1}/{retries}...")
                
                response = completion(**completion_args)
                content = response["choices"][0]["message"]["content"]
                
                # If no response model, return raw content
                if self.raw:
                    print(f"✓ Success on attempt {attempt + 1}\n")
                    return content
                
                # Parse structured output
                cleaned_content = self._clean_json_response(content)
                result = self.response_model.model_validate_json(cleaned_content)
                
                print(f"✓ Success on attempt {attempt + 1}\n")
                return result
                
            except Exception as e:
                last_error = e
                print(f"✗ Error on attempt {attempt + 1}: {e}")
                
                # Wait before retry
                if attempt < retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...\n")
                    time.sleep(self.retry_delay)
        
        # All retries failed
        raise Exception(f"Failed after {retries} attempts. Last error: {last_error}")
    

