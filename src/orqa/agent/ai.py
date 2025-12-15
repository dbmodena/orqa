import os 
from litellm import completion
import importlib
from pydantic import BaseModel, ValidationError
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
        """Clean up JSON response by extracting valid JSON"""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        # Try to find JSON object or array
        # Look for content between outermost { } or [ ]
        brace_start = content.find('{')
        bracket_start = content.find('[')
        
        # Determine which comes first
        if brace_start == -1:
            start = bracket_start
        elif bracket_start == -1:
            start = brace_start
        else:
            start = min(brace_start, bracket_start)
        
        if start == -1:
            return content
        
        # Find matching closing character
        if content[start] == '{':
            # Find the last closing brace
            depth = 0
            for i in range(start, len(content)):
                if content[i] == '{':
                    depth += 1
                elif content[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return content[start:i+1].strip()
        else:
            # Find the last closing bracket
            depth = 0
            for i in range(start, len(content)):
                if content[i] == '[':
                    depth += 1
                elif content[i] == ']':
                    depth -= 1
                    if depth == 0:
                        return content[start:i+1].strip()
        
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


    def _format_json_error(self, content: str, error: Exception) -> str:
        """
        Format JSON parsing error with context.
        
        Args:
            content: The content that failed to parse
            error: The JSON parsing exception
            
        Returns:
            Human-readable error message for the LLM
        """
        # Show a snippet of the problematic content
        snippet = content[:200] + "..." if len(content) > 200 else content
        
        formatted_error = (
            "❌ JSON PARSING ERROR - Your response is not valid JSON.\n\n"
            f"Error: {str(error)}\n\n"
            #f"Your response (first 200 chars):\n{snippet}\n\n"
            "Required schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "⚠️ Common issues:\n"
            "  • Missing quotes around strings\n"
            "  • Trailing commas\n"
            "  • Unescaped special characters\n"
            "  • Text before or after the JSON object\n\n"
            "Please generate ONLY a valid JSON object matching the schema above."
        )
        
        return formatted_error
    
    def _format_validation_error(self, error: ValidationError) -> str:
        """
        Format Pydantic validation error in a clear, actionable way.
        
        Args:
            error: Pydantic ValidationError
            
        Returns:
            Human-readable error message for the LLM
        """
        error_messages = []
        
        for err in error.errors():
            field_path = " -> ".join(str(x) for x in err['loc'])
            error_type = err['type']
            message = err['msg']
            
            error_messages.append(
                f"  • Field '{field_path}': {message} (error type: {error_type})"
            )
        
        formatted_error = (
            "❌ VALIDATION ERROR - Your JSON response does not match the required schema.\n\n"
            "Required schema:\n"
            f"{json.dumps(self.response_model.model_json_schema(), indent=2)}\n\n"
            "Validation errors found:\n" +
            "\n".join(error_messages) +
            "\n\n⚠️ Please fix these issues and generate a valid JSON response."
        )
        
        return formatted_error

    def complete(
        self, prompt:str,
        reply_model: Optional[Type[BaseModel]] = None,
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
        messages = [
            {"role": "system", "content": prompt}
        ]
        completion_args = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            **kwargs
        }
        # Enable JSON mode if we have a response model
        if reply_model and self.enable_json_mode:
            completion_args["response_format"] = {"type": "json_object"}
            self.response_model = reply_model
            self.raw = False    
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
                last_content = content
                cleaned_content = self._clean_json_response(content)
                print(cleaned_content)
                try:
                    # First try to parse as JSON
                    json_data = json.loads(cleaned_content)
                    # Then validate with Pydantic
                    result = self.response_model.model_validate(json_data)
                    print(f"✓ Success on attempt {attempt + 1}\n")
                    return result.model_dump()
                except json.JSONDecodeError as e:
                    # JSON parsing failed
                    last_error = e
                    error_msg = self._format_json_error(cleaned_content, e)
                    print(f"⚠️ JSON parsing error on attempt {attempt + 1}")
                    
                    if attempt < retries - 1:
                        # Add assistant's failed response
                        messages.append({
                            "role": "assistant",
                            "content": content
                        })
                        # Add error feedback as user message
                        messages.append({
                            "role": "user",
                            "content": error_msg
                        })
                        print(f"💬 Sending error feedback to LLM...\n")
                        time.sleep(self.retry_delay)
                        continue
                except ValidationError as e:
                    # Pydantic validation failed
                    last_error = e
                    error_msg = self._format_validation_error(e)
                    print(f"⚠️ Validation error on attempt {attempt + 1}")
                    
                    if attempt < retries - 1:
                        # Add assistant's failed response
                        messages.append({
                            "role": "assistant",
                            "content": content
                        })
                        # Add error feedback as user message
                        messages.append({
                            "role": "user",
                            "content": error_msg
                        })
                        print(f"💬 Sending validation errors to LLM...\n")
                        time.sleep(self.retry_delay)
                        continue
                
            except Exception as e:
                last_error = e
                print(f"✗ Error on attempt {attempt + 1}: {e}")
                
                # Wait before retry
                if attempt < retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...\n")
                    time.sleep(self.retry_delay)
        
        # All retries failed
        # All retries exhausted
        print(f"\n❌ Failed after {retries} attempts")
        print(f"Last error: {last_error}")
        if last_content and not self.raw:
            print(f"\nLast response preview:\n{last_content[:300]}...\n")
        
        return self.response_model().model_dump_json(indent=2)
                

        
        
    


    

