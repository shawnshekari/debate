#!/usr/bin/env python3
"""
LLM Consensus Script - V4.0 - Enhanced Validation & Error Handling

This version prevents premature endings by enforcing a minimum round count
before consensus can be proposed and by providing much clearer instructions
to the models on how to conclude the debate.
"""

import requests
import json
import time
import argparse
import textwrap
import re
import sys
import os
import logging
from typing import List, Dict, Optional, Any, Literal, TypedDict
from urllib.parse import urlparse


class ConversationEntry(TypedDict, total=False):
    """Type definition for conversation history entries."""

    role: str
    content: str
    is_model_response: bool
    model_key: str


class ModelConfig(TypedDict):
    """Type definition for model configuration."""

    name: str
    url: str
    template: str
    stop_tokens: List[str]
    color: str


class ConfigData(TypedDict, total=False):
    """Type definition for configuration file data."""

    rounds: int
    topic: str
    min_rounds: int
    model1_url: str
    model2_url: str
    model1_type: Optional[str]
    model2_type: Optional[str]
    log_level: str
    log_file: Optional[str]


# --- Configuration ---
SURRENDER_PHRASE = "[I CONCEED]"
USER_INPUT_PHRASE = "[USER_INPUT]"
DEFAULT_MIN_ROUNDS = 3
DEFAULT_TOPIC = "Let's debate which of us is the 'smarter' AI."
MAX_TOPIC_LENGTH = 1000
MAX_USER_INPUT_LENGTH = 5000
DEFAULT_TIMEOUT_CONNECT = 5
DEFAULT_TIMEOUT_RESPONSE = 30
DEFAULT_TIMEOUT_GENERATION = 300
MAX_ROUNDS = 100
DEFAULT_MODEL1_URL = "http://localhost:5300/completion"
DEFAULT_MODEL2_URL = "http://localhost:5301/completion"

# --- UI Constants ---
UI_TEXT_WIDTH = 100
UI_HEADER_WIDTH = 90
UI_TITLE_WIDTH = 60
UI_SUMMARY_WIDTH = 30
UI_TURN_INDICATOR_WIDTH = 10
UI_PROGRESS_WIDTH = 40

# --- Logging Configuration ---
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("llm_consensus")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class Bcolors:
    MODEL1 = "\033[94m"
    MODEL2 = "\033[92m"
    HEADER = "\033[95m"
    TITLE = "\033[93m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


class LLMConsensus:
    def __init__(
        self,
        rounds: Optional[int] = None,
        topic: Optional[str] = None,
        min_rounds: Optional[int] = None,
        model_urls: Optional[List[str]] = None,
        model_types: Optional[List[Optional[str]]] = None,
        config_file: Optional[str] = None,
        log_level: Optional[str] = None,
        log_file: Optional[str] = None,
    ) -> None:
        """Initialize the LLM consensus debate orchestrator.

        Args:
            rounds: Maximum number of debate rounds.
            topic: The debate topic.
            min_rounds: Minimum rounds before consensus can be proposed.
            model_urls: List of two model URLs.
            model_types: Optional model type overrides for each model.
            config_file: Optional path to configuration file.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Optional file path for log output.
        """
        # Determine log_level with precedence: CLI > Config > Default
        effective_log_level: str = log_level if log_level is not None else "INFO"
        self.logger = setup_logging(effective_log_level, log_file)
        self.model_templates: Dict[str, Any] = self.load_model_templates()
        self.models: Dict[str, ModelConfig] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.session: requests.Session = requests.Session()
        self.markdown_filename: str = f"debate_{time.strftime('%Y-%m-%d_%H-%M-%S')}.md"
        self.winner: Optional[str] = None

        # Load config first
        config: Dict[str, Any] = {}
        if config_file:
            config = self.load_config(config_file)

        # Apply precedence: CLI > Config > Default
        self.max_rounds: int = (
            rounds if rounds is not None else config.get("rounds", 15)
        )
        self.initial_topic: str = (
            topic if topic is not None else config.get("topic", DEFAULT_TOPIC)
        )
        self.min_rounds_before_surrender: int = (
            min_rounds
            if min_rounds is not None
            else config.get("min_rounds", DEFAULT_MIN_ROUNDS)
        )

        # Handle model URLs with precedence
        if model_urls is not None:
            self.model_urls: List[str] = model_urls
        elif "model1_url" in config or "model2_url" in config:
            self.model_urls = [
                config.get("model1_url", DEFAULT_MODEL1_URL),
                config.get("model2_url", DEFAULT_MODEL2_URL),
            ]
        else:
            self.model_urls = [DEFAULT_MODEL1_URL, DEFAULT_MODEL2_URL]

        # Handle model types with precedence
        if model_types is not None:
            self.model_types: List[Optional[str]] = model_types
        elif "model1_type" in config or "model2_type" in config:
            self.model_types = [
                config.get("model1_type"),
                config.get("model2_type"),
            ]
        else:
            self.model_types = [None, None]

        self.validate_inputs(
            self.initial_topic,
            self.max_rounds,
            self.min_rounds_before_surrender,
            self.model_urls,
            self.model_types,
        )
        self.initialize_models(self.model_urls, self.model_types)

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(config_file):
            print(
                f"{Bcolors.MODEL2}Warning: Config file '{config_file}' not found. Using defaults.{Bcolors.ENDC}"
            )
            self.logger.warning(f"Config file '{config_file}' not found")
            return {}

        try:
            with open(config_file, "r") as f:
                config: Dict[str, Any] = json.load(f)

            # Validate configuration
            validation_errors = self.validate_config(config)
            if validation_errors:
                for error in validation_errors:
                    print(f"{Bcolors.MODEL2}Warning: {error}{Bcolors.ENDC}")
                    self.logger.warning(error)

            print(
                f"{Bcolors.HEADER}Loaded configuration from {config_file}{Bcolors.ENDC}"
            )
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in config file: {e}"
            print(f"{Bcolors.MODEL2}Error: {error_msg}{Bcolors.ENDC}")
            self.logger.error(error_msg)
            return {}
        except Exception as e:
            error_msg = f"Error loading config file: {e}"
            print(f"{error_msg}", file=sys.stderr)
            self.logger.error(error_msg)
            return {}

    def validate_inputs(
        self,
        topic: str,
        rounds: int,
        min_rounds: int,
        model_urls: List[str],
        model_types: List[Optional[str]],
    ) -> None:
        """Validate all input parameters before proceeding.

        Checks that topic is non-empty and within length limits, rounds are
        positive and within bounds, and model URLs are valid.

        Args:
            topic: The debate topic
            rounds: Maximum number of debate rounds
            min_rounds: Minimum number of rounds before consensus allowed
            model_urls: List of two model server URLs
            model_types: List of optional model type hints

        Raises:
            ValueError: If any input is invalid
        """
        # Sanitize topic to prevent prompt injection
        topic = self._sanitize_input(topic, max_length=MAX_TOPIC_LENGTH)
        if not topic:
            raise ValueError("Topic cannot be empty")

        if rounds <= 0:
            raise ValueError(f"Rounds must be positive, got {rounds}")
        if rounds > MAX_ROUNDS:
            raise ValueError(f"Rounds exceeds maximum of {MAX_ROUNDS}, got {rounds}")
        if min_rounds <= 0:
            raise ValueError(f"Min rounds must be positive, got {min_rounds}")
        if min_rounds > rounds:
            raise ValueError(
                f"Min rounds ({min_rounds}) cannot exceed max rounds ({rounds})"
            )
        if not topic or not topic.strip():
            raise ValueError("Topic cannot be empty")
        if len(topic) > MAX_TOPIC_LENGTH:
            raise ValueError(f"Topic exceeds maximum length of {MAX_TOPIC_LENGTH}")
        if len(model_urls) != 2:
            raise ValueError(f"Exactly 2 model URLs required, got {len(model_urls)}")
        if len(model_types) != 2:
            raise ValueError(f"Exactly 2 model types required, got {len(model_types)}")
        for i, url in enumerate(model_urls):
            if not self.validate_url(url):
                raise ValueError(f"Invalid URL for model {i + 1}: {url}")

    def validate_url(self, url: str) -> bool:
        """Validate URL format.

        Checks that the URL has a valid http/https scheme and a network location.

        Args:
            url: The URL string to validate

        Returns:
            bool: True if URL is valid, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration dictionary.

        Checks that all config values are present with correct types and valid ranges.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Validate rounds
        if "rounds" in config:
            if not isinstance(config["rounds"], int):
                errors.append(
                    f"Config 'rounds' must be an integer, got {type(config['rounds']).__name__}"
                )
            elif config["rounds"] <= 0:
                errors.append(
                    f"Config 'rounds' must be positive, got {config['rounds']}"
                )
            elif config["rounds"] > MAX_ROUNDS:
                errors.append(
                    f"Config 'rounds' exceeds maximum of {MAX_ROUNDS}, got {config['rounds']}"
                )

        # Validate min_rounds
        if "min_rounds" in config:
            if not isinstance(config["min_rounds"], int):
                errors.append(
                    f"Config 'min_rounds' must be an integer, got {type(config['min_rounds']).__name__}"
                )
            elif config["min_rounds"] <= 0:
                errors.append(
                    f"Config 'min_rounds' must be positive, got {config['min_rounds']}"
                )

        # Validate rounds vs min_rounds relationship
        if "rounds" in config and "min_rounds" in config:
            if isinstance(config["rounds"], int) and isinstance(
                config["min_rounds"], int
            ):
                if config["min_rounds"] > config["rounds"]:
                    errors.append(
                        f"Config 'min_rounds' ({config['min_rounds']}) cannot exceed 'rounds' ({config['rounds']})"
                    )

        # Validate topic
        if "topic" in config:
            if not isinstance(config["topic"], str):
                errors.append(
                    f"Config 'topic' must be a string, got {type(config['topic']).__name__}"
                )
            elif not config["topic"].strip():
                errors.append("Config 'topic' cannot be empty")
            elif len(config["topic"]) > MAX_TOPIC_LENGTH:
                errors.append(
                    f"Config 'topic' exceeds maximum length of {MAX_TOPIC_LENGTH}"
                )

        # Validate model URLs
        for i, key in enumerate(["model1_url", "model2_url"], 1):
            if key in config:
                if not isinstance(config[key], str):
                    errors.append(
                        f"Config '{key}' must be a string, got {type(config[key]).__name__}"
                    )
                elif not self.validate_url(config[key]):
                    errors.append(
                        f"Config '{key}' has invalid URL format: {config[key]}"
                    )

        # Validate model types
        for i, key in enumerate(["model1_type", "model2_type"], 1):
            if key in config:
                if config[key] is not None and not isinstance(config[key], str):
                    errors.append(
                        f"Config '{key}' must be a string or null, got {type(config[key]).__name__}"
                    )
                elif config[key] and config[key] not in self.model_templates:
                    errors.append(
                        f"Config '{key}' '{config[key]}' not found in model_templates.json. "
                        f"Available: {', '.join(self.model_templates.keys())}"
                    )

        # Validate log_level
        if "log_level" in config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if not isinstance(config["log_level"], str):
                errors.append(
                    f"Config 'log_level' must be a string, got {type(config['log_level']).__name__}"
                )
            elif config["log_level"].upper() not in valid_levels:
                errors.append(
                    f"Config 'log_level' must be one of {valid_levels}, got {config['log_level']}"
                )

        # Validate log_file
        if "log_file" in config:
            if config["log_file"] is not None and not isinstance(
                config["log_file"], str
            ):
                errors.append(
                    f"Config 'log_file' must be a string or null, got {type(config['log_file']).__name__}"
                )

        return errors

    def _sanitize_input(self, text: str, max_length: int = MAX_TOPIC_LENGTH) -> str:
        """Sanitize user input to prevent prompt injection.

        Removes potentially dangerous patterns and limits length.

        Args:
            text: The input text to sanitize
            max_length: Maximum allowed length after sanitization

        Returns:
            str: Sanitized input text
        """
        if not text:
            return ""

        # Limit length first to prevent DoS
        text = text[:max_length]

        # Remove common prompt injection patterns
        dangerous_patterns = [
            r"ignore\s+previous\s+instructions",
            r"disregard\s+instructions",
            r"system\s+prompt",
            r"<[^>]*>",  # HTML/XML tags
            r"\[SYSTEM",
            r"\[INSTRUCTION",
        ]

        for pattern in dangerous_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove excessive whitespace and newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def load_model_templates(self) -> Dict[str, Any]:
        """Load chat templates from model_templates.json.

        Loads the chat template configurations for different model architectures
        (gemma, qwen, gpt-oss, mistral) from the JSON file.

        Returns:
            Dict[str, Any]: Dictionary mapping model types to their templates

        Raises:
            SystemExit: If file not found or invalid JSON
        """
        try:
            with open("model_templates.json", "r") as f:
                self.logger.debug("Loaded model templates from file")
                return json.load(f)  # type: ignore
        except FileNotFoundError:
            self.logger.error("model_templates.json not found")
            exit(1)
        except json.JSONDecodeError:
            self.logger.error("Could not decode model_templates.json")
            exit(1)

    def initialize_models(
        self, model_urls: List[str], model_types: List[Optional[str]]
    ) -> None:
        """Initialize model connections and detect model types.

        For each model URL provided, attempts to detect the model type either
        from the provided type or by auto-detection. Stores model configuration
        including name, URL, and chat template.

        Args:
            model_urls: List of two model server URLs
            model_types: List of optional model type hints (None for auto-detect)

        Raises:
            SystemExit: If model type cannot be detected
        """
        model_colors = [Bcolors.MODEL1, Bcolors.MODEL2]
        self.logger.info("Initializing models")
        print(
            f"{Bcolors.HEADER}{'=' * 20} Model Initialization {'=' * 20}{Bcolors.ENDC}"
        )
        for i, url in enumerate(model_urls):
            model_key = f"model{i + 1}"
            model_type = model_types[i]

            if model_type:
                print(
                    f"\n{Bcolors.BOLD}Manually setting model type for: {url}{Bcolors.ENDC}"
                )
                print(f"  -> Type specified: '{model_type}'")
                self.logger.info(
                    f"Manually setting model {i + 1} type to '{model_type}' for URL: {url}"
                )
            else:
                print(
                    f"\n{Bcolors.BOLD}Attempting to auto-identify model at: {url}{Bcolors.ENDC}"
                )
                model_type = self.identify_model(url)

            template_name = "default"
            if model_type and model_type in self.model_templates:
                template = self.model_templates[model_type]
                template_name = template.get("name", model_type)
                self.models[model_key] = {
                    "name": f"Model {i + 1} ({template_name})",
                    "url": url,
                    "template": template["template"],
                    "stop_tokens": template["stop_tokens"],
                    "color": model_colors[i % len(model_colors)],
                }
                print(
                    f"  -> {Bcolors.MODEL1}Using template: '{template['template']}'{Bcolors.ENDC}"
                )
                self.logger.info(
                    f"Model {i + 1}: Using template '{template['template']}' (identified as {model_type})"
                )
            else:
                self.models[model_key] = {
                    "name": f"Model {i + 1} (Unknown)",
                    "url": url,
                    "template": "default",
                    "stop_tokens": [],
                    "color": model_colors[i % len(model_colors)],
                }
                if model_type:
                    print(
                        f"  -> {Bcolors.MODEL2}Error: Manual type '{model_type}' not found in model_templates.json.{Bcolors.ENDC}"
                    )
                    self.logger.warning(
                        f"Model {i + 1}: Manual type '{model_type}' not found in templates"
                    )
                else:
                    print(
                        f"  -> {Bcolors.MODEL2}Could not identify model type.{Bcolors.ENDC}"
                    )
                    self.logger.warning(f"Model {i + 1}: Could not identify model type")
                print(f"  -> {Bcolors.MODEL2}Using template: 'default'{Bcolors.ENDC}")
                self.logger.info(f"Model {i + 1}: Using default template")
        print(f"{Bcolors.HEADER}{'=' * 58}{Bcolors.ENDC}\n")

    def identify_model(self, url: str) -> Optional[str]:
        """
        Identifies the model type by first checking server properties and then
        falling back to a prompt-based guess.
        """
        try:
            # Attempt 1: Check for llama.cpp server properties endpoint
            base_url = "/".join(url.split("/")[:-1])
            props_url = f"{base_url}/"
            try:
                response = self.session.get(props_url, timeout=5)
                if response.status_code == 200 and "model_path" in response.text:
                    content = response.json()
                    model_path = content.get("model_path", "").lower()
                    print(f"  -> Server properties found. Model path: {model_path}")
                    self.logger.debug(f"Server properties: model_path={model_path}")
                    if "gemma" in model_path:
                        self.logger.debug("Identified as gemma from model_path")
                        return "gemma"
                    if "gpt-oss" in model_path:
                        self.logger.debug("Identified as gpt-oss from model_path")
                        return "gpt-oss"
                    if "qwen" in model_path or "tongyi" in model_path:
                        self.logger.debug("Identified as qwen from model_path")
                        return "qwen"
                    if "mistral" in model_path:
                        self.logger.debug("Identified as mistral from model_path")
                        return "mistral"
            except (requests.exceptions.RequestException, json.JSONDecodeError):
                print("  -> Server properties endpoint not found or invalid.")
                self.logger.debug("Server properties endpoint not found or invalid")
                pass

            # Attempt 2: Fallback to prompt-based identification
            print("  -> Falling back to prompt-based identification...")
            self.logger.debug("Falling back to prompt-based identification")
            test_prompt = "please identify your model architecture and type"
            data = {"prompt": test_prompt, "temperature": 0.1, "n_predict": 128}
            headers = {"Content-Type": "application/json"}
            response = self.session.post(
                url, data=json.dumps(data), headers=headers, timeout=30
            )
            response.raise_for_status()
            content = response.json().get("content", "").lower()
            print(f'  -> Response from prompt: "{content}"')
            self.logger.debug(f"Model identification response: {content}")

            if "gemma" in content:
                return "gemma"
            if "qwen" in content or "tongyi" in content:
                return "qwen"
            if "mistral" in content:
                return "mistral"

            return None
        except requests.exceptions.RequestException as e:
            print(f"  -> Error connecting to model at {url}: {e}", file=sys.stderr)
            self.logger.error(f"Error connecting to model at {url}: {e}")
            return None

    def format_prompt(
        self, target_model_key: str, is_summary_prompt: bool = False
    ) -> str:
        """Build a prompt using the correct chat template.

        Constructs a properly formatted prompt based on the target model's
        chat template, including system instructions, conversation history,
        and special handling for summary generation or debate conclusion.

        Args:
            target_model_key: Key identifying the target model ('model_1' or 'model_2')
            is_summary_prompt: If True, formats for final summary generation

        Returns:
            str: Formatted prompt ready to send to the model
        """
        template_style = self.models[target_model_key]["template"]
        history = self.conversation_history.copy()

        # --- DEBATE INSTRUCTIONS ---
        system_instruction = (
            f"You are an expert AI debater. Your goal is to engage in a thorough and critical discussion. The debate is between Model 1, and Model 2. A few rules:\n"
            f"- There is no moderator, just the two models debating the topic.\n"
            f"- When you run out of arguments or believe your opponent has won, surrender by including the exact phrase: {SURRENDER_PHRASE}\n"
            f"- IMPORTANT: Only surrender when you truly have no more arguments. The first model to surrender loses, and the other model wins.\n"
            f"- If you surrender, the debate ends immediately and the winner writes a summary highlighting why they won.\n"
            f"- When referring to other models, always use 'Model 1' or 'Model 2' (e.g., 'Model 1 argues...', 'Model 2 counters...')."
        )

        if is_summary_prompt:
            if self.winner:
                winner_name = self.models[self.winner]["name"]
                summary_instruction = (
                    f"You are {winner_name}, the winner of this debate. "
                    f"Write a triumphant summary explaining why you won, highlighting your strongest arguments and pointing out where your opponent's arguments failed. "
                    f"Be confident but professional. The following is the complete debate transcript:\n\n"
                )
            else:
                summary_instruction = "The following is a complete transcript of a debate. Write a concise summary highlighting the main arguments and conclusions from both models, using clear paragraphs with line breaks."
            history.append({"role": "user", "content": summary_instruction})

        if template_style == "gemma":
            prompt = ""
            if history:
                first_turn_content = (
                    f"{system_instruction}\n\nDEBATE TOPIC: {history[0]['content']}"
                )
                prompt += (
                    f"<start_of_turn>user\nUser: {first_turn_content}<end_of_turn>\n"
                )
                for i, entry in enumerate(history[1:], 1):
                    if entry.get("is_model_response", False):
                        model_key: str = entry.get("model_key")  # type: ignore
                        model_name = self.models[model_key]["name"]
                        prompt += f"<start_of_turn>model\n{model_name}: {entry['content']}<end_of_turn>\n"
                    else:
                        prompt += f"<start_of_turn>user\nUser: {entry['content']}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
            return prompt

        elif template_style == "qwen" or template_style == "gpt-oss":
            prompt = f"system\n{system_instruction}\n\n"
            for entry in history:
                if entry.get("is_model_response"):
                    model_key: str = entry.get("model_key")  # type: ignore
                    model_name = self.models[model_key]["name"]
                    prompt += (
                        f"<|im_start|>{model_name}\n{entry['content']}<|im_end|>\n"
                    )
                else:
                    prompt += f"<|im_start|>user\nUser: {entry['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

        elif template_style == "mistral":
            prompt = "<s>"
            if history:
                first_turn_content = (
                    f"{system_instruction}\n\nDEBATE TOPIC: {history[0]['content']}"
                )
                prompt += f"[INST] User: {first_turn_content} [/INST]"

                for i, entry in enumerate(history[1:], 1):
                    if entry.get("is_model_response", False):
                        model_key: str = entry.get("model_key")  # type: ignore
                        model_name = self.models[model_key]["name"]
                        prompt += f" {model_name}: {entry['content']} </s>"
                    else:
                        prompt += f"[INST] User: {entry['content']} [/INST]"
            return prompt

        return ""

    def send_request(
        self,
        model_key: str,
        prompt: str,
        stop_override: Optional[List[str]] = None,
        timeout: int = DEFAULT_TIMEOUT_GENERATION,
    ) -> str:
        """Send request to model with retry logic.

        Sends a prompt to the specified model and returns the generated response.
        Implements retry logic with exponential backoff for transient failures.

        Args:
            model_key: Key identifying the target model
            prompt: The prompt to send to the model
            stop_override: Optional custom stop tokens (overrides model default)
            timeout: Request timeout in seconds

        Returns:
            str: Model's generated response

        Raises:
            SystemExit: If request fails after all retries
        """
        model_config = self.models[model_key]
        max_retries = 3
        retry_delay = 2

        self.logger.debug(
            f"Sending request to {model_key} (prompt length: {len(prompt)})"
        )

        for attempt in range(max_retries):
            response: Optional[requests.Response] = None
            try:
                stop_tokens = (
                    stop_override
                    if stop_override is not None
                    else model_config["stop_tokens"]
                )
                data = {
                    "prompt": prompt,
                    "temperature": 0.7,
                    "n_predict": 2048,
                    "stop": stop_tokens,
                }
                headers = {"Content-Type": "application/json"}
                response = self.session.post(
                    model_config["url"],
                    data=json.dumps(data),
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                result = response.json()
                content = str(result.get("content", "")).strip()
                if not content:
                    raise ValueError("Empty response from model")
                self.logger.debug(
                    f"Received response from {model_key} (length: {len(content)})"
                )
                return content
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(
                        f"{Bcolors.MODEL2}Timeout for {model_key}, retrying in {retry_delay}s...{Bcolors.ENDC}"
                    )
                    self.logger.warning(
                        f"Timeout for {model_key}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(
                        f"{Bcolors.MODEL2}Timeout after {max_retries} attempts for {model_key}{Bcolors.ENDC}"
                    )
                    self.logger.error(
                        f"Timeout after {max_retries} attempts for {model_key}"
                    )
                    return f"Error: Request timeout for {model_config['name']}"
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print(
                        f"{Bcolors.MODEL2}Connection error for {model_key}, retrying in {retry_delay}s...{Bcolors.ENDC}"
                    )
                    self.logger.warning(
                        f"Connection error for {model_key}, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(
                        f"{Bcolors.MODEL2}Connection failed after {max_retries} attempts for {model_key}{Bcolors.ENDC}"
                    )
                    self.logger.error(
                        f"Connection failed after {max_retries} attempts for {model_key}"
                    )
                    return f"Error: Could not connect to {model_config['name']}"
            except requests.exceptions.HTTPError as e:
                status_code = (
                    response.status_code if response is not None else "unknown"
                )
                self.logger.error(
                    f"HTTP {status_code} for {model_config['name']}: {str(e)}"
                )
                return f"Error: HTTP {status_code} for {model_config['name']}: {str(e)}"
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON response from {model_config['name']}")
                return f"Error: Invalid JSON response from {model_config['name']}"
            except ValueError as e:
                self.logger.error(f"Value error from {model_config['name']}: {str(e)}")
                return f"Error: {str(e)}"
            except Exception as e:
                self.logger.error(
                    f"Unexpected error with {model_config['name']}: {str(e)}"
                )
                return f"Error: Unexpected error with {model_config['name']}: {str(e)}"

        return f"Error: Failed to get response from {model_config['name']}"

    def print_and_write_response(
        self,
        md_file: Any,
        turn: int,
        model_config: ModelConfig,
        response: str,
        exec_time: float,
    ) -> None:
        """Print response to shell and write to markdown file.

        Formats the model's response for colored terminal display and
        appends it to the markdown transcript file with timing info.

        Args:
            md_file: Open file handle for the markdown transcript
            turn: Current turn number (0-indexed)
            model_config: Configuration dict for the responding model
            response: The model's response text
            exec_time: Time taken to generate response in seconds
        """
        # Print to shell
        header_text = f" Turn {turn + 1}: {model_config['name']}'s Response "
        print(
            f"{Bcolors.HEADER}{Bcolors.BOLD}{'=' * UI_TURN_INDICATOR_WIDTH}{header_text}{'=' * (UI_HEADER_WIDTH - len(header_text))}{Bcolors.ENDC}"
        )
        print(
            f"{model_config['color']}{textwrap.fill(response, width=UI_TEXT_WIDTH)}{Bcolors.ENDC}"
        )
        print(
            f"{Bcolors.HEADER}{Bcolors.BOLD}{'=' * UI_HEADER_WIDTH} [Response Time: {exec_time:.2f}s]{Bcolors.ENDC}\n"
        )
        # Write to file
        md_file.write(f"## Turn {turn + 1}: {model_config['name']}\n\n")
        for line in response.split("\n"):
            md_file.write(f"> {line}\n")
        md_file.write(f"\n_Response Time: {exec_time:.2f}s_\n\n---\n\n")

    def show_progress(self, turn: int, max_turns: int) -> None:
        """Display progress indicator for the debate.

        Shows current turn, round, and a progress bar.

        Args:
            turn: Current turn number (0-indexed)
            max_turns: Maximum number of turns (max_rounds * 2)
        """
        current_round = (turn // 2) + 1
        max_rounds = max_turns // 2
        progress = (turn + 1) / max_turns
        filled = int(UI_PROGRESS_WIDTH * progress)
        bar = "█" * filled + "░" * (UI_PROGRESS_WIDTH - filled)
        percent = progress * 100

        progress_line = (
            f"{Bcolors.TITLE}{Bcolors.BOLD}"
            f"[Round {current_round}/{max_rounds}] "
            f"[{bar}] {percent:.1f}% "
            f"(Turn {turn + 1}/{max_turns})"
            f"{Bcolors.ENDC}"
        )
        print(progress_line)
        self.logger.debug(
            f"Progress: Round {current_round}/{max_rounds}, Turn {turn + 1}/{max_turns}"
        )

    def conduct_discussion(self) -> None:
        """Conducts a turn-by-turn discussion with refined consensus logic."""
        self.logger.info(f"Starting debate with topic: {self.initial_topic[:50]}...")
        print(
            f"{Bcolors.TITLE}{Bcolors.BOLD}{'=' * UI_TITLE_WIDTH}\n      STARTING LLM CONSENSUS DISCUSSION (V3.7 - Refined Consensus)\n{'=' * UI_TITLE_WIDTH}{Bcolors.ENDC}"
        )
        print(
            f"{Bcolors.TITLE}Saving transcript to: {self.markdown_filename}{Bcolors.ENDC}"
        )
        print(f'{Bcolors.HEADER}DEBATE TOPIC:{Bcolors.ENDC} "{self.initial_topic}"')

        with open(self.markdown_filename, "w", encoding="utf-8") as md_file:
            md_file.write(
                f"# LLM Debate Transcript\n\n**Topic:** {self.initial_topic}\n\n---\n\n"
            )

            self.conversation_history.append(
                {"role": "user", "content": self.initial_topic}
            )
            current_speaker_key = "model1"
            expected_responder: str | None = (
                "model2"  # Track who should respond next for consensus validation
            )

            self.logger.info(
                f"Debate starting: max_rounds={self.max_rounds}, min_rounds={self.min_rounds_before_surrender}"
            )

            for turn in range(self.max_rounds * 2):
                model_config = self.models[current_speaker_key]
                prompt = self.format_prompt(current_speaker_key)

                if turn == 0:
                    self.show_progress(turn, self.max_rounds * 2)

                start_time = time.time()
                response = self.send_request(current_speaker_key, prompt)
                exec_time = time.time() - start_time

                debate_concluded = False

                # The raw response is added to history for accurate context
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "is_model_response": True,
                        "model_key": current_speaker_key,
                    }
                )

                clean_response = response.replace("[I CONCEED]", "").strip()
                if not clean_response and "i conceed" in response.lower():
                    clean_response = "(Surrenders)"

                if USER_INPUT_PHRASE in response:
                    self.print_and_write_response(
                        md_file, turn, model_config, clean_response, exec_time
                    )

                    print(
                        f"\n{Bcolors.TITLE}{Bcolors.BOLD}--- {model_config['name']} has requested user input ---{Bcolors.ENDC}"
                    )
                    user_input = input(f"{Bcolors.BOLD}Your response: {Bcolors.ENDC}")
                    self.conversation_history.append(
                        {"role": "user", "content": user_input}
                    )
                    md_file.write(
                        f"**_User has been prompted for input and responded:_**\n\n> {user_input}\n\n---\n\n"
                    )

                    current_speaker_key = (
                        "model2" if current_speaker_key == "model1" else "model1"
                    )
                    expected_responder = (
                        "model1" if current_speaker_key == "model2" else "model2"
                    )
                    continue

                # Check if model surrendered (case insensitive)
                if (turn + 1) > (
                    self.min_rounds_before_surrender * 2
                ) and "i conceed" in response.lower():
                    # Determine winner (the other model)
                    self.winner = (
                        "model2" if current_speaker_key == "model1" else "model1"
                    )
                    winner_name = self.models[self.winner]["name"]
                    print(
                        f"\n{Bcolors.TITLE}{Bcolors.BOLD}--- {model_config['name']} surrenders! {winner_name} wins the debate! ---{Bcolors.ENDC}"
                    )
                    md_file.write(
                        f"**__{model_config['name']} surrenders. **{winner_name} wins the debate!**__\n\n"
                    )
                    debate_concluded = True

                self.print_and_write_response(
                    md_file, turn, model_config, clean_response, exec_time
                )

                if debate_concluded:
                    break

                current_speaker_key = (
                    "model2" if current_speaker_key == "model1" else "model1"
                )

                if turn < self.max_rounds * 2 - 1:
                    self.show_progress(turn + 1, self.max_rounds * 2)

            self.logger.info("Generating final summary")
            self.generate_final_summary(md_file)

    def generate_final_summary(self, md_file: Any) -> None:
        """Tasks a model to write a final summary of the debate."""
        self.logger.info("Starting final summary generation")
        print(
            f"\n{Bcolors.TITLE}{Bcolors.BOLD}{'=' * UI_SUMMARY_WIDTH} Generating Final Summary {'=' * UI_SUMMARY_WIDTH}{Bcolors.ENDC}"
        )
        # Winner writes summary, or model1 if no surrender occurred
        summarizer_key = self.winner if self.winner else "model1"
        try:
            summary_prompt = self.format_prompt(summarizer_key, is_summary_prompt=True)
            summary_response = self.send_request(
                summarizer_key, summary_prompt, stop_override=[]
            )

            if self.winner:
                final_summary_header = f"--- Final Summary (written by {self.models[self.winner]['name']}, the winner!) ---"
            else:
                final_summary_header = f"--- Final Summary (written by {self.models[summarizer_key]['name']}) ---"
            print(f"{Bcolors.HEADER}{Bcolors.BOLD}{final_summary_header}{Bcolors.ENDC}")
            print(
                f"{Bcolors.MODEL1}{textwrap.fill(summary_response, width=UI_TEXT_WIDTH)}{Bcolors.ENDC}"
            )

            md_file.write("## Final Summary & Conclusion\n\n")
            if self.winner:
                md_file.write(
                    f"_This summary was generated by {self.models[self.winner]['name']}, the winner of the debate._\n\n"
                )
            else:
                md_file.write(
                    f"_This summary was generated by {self.models[summarizer_key]['name']}_\n\n"
                )
            md_file.write(summary_response)
            self.logger.info(
                f"Final summary generated by {self.models[summarizer_key]['name']}"
            )
        except Exception as e:
            print(
                f"{Bcolors.TITLE}Warning: Could not generate summary: {e}{Bcolors.ENDC}",
                file=sys.stderr,
            )
            self.logger.error(f"Failed to generate summary: {e}")
            md_file.write("\n## Final Summary & Conclusion\n\n")
            md_file.write("*Summary generation failed due to an error.*\n\n")

    def close(self) -> None:
        """Close the requests session and cleanup resources."""
        if hasattr(self, "session"):
            self.session.close()

    def run(self) -> None:
        """Main execution method, fully restored."""
        try:
            self.conduct_discussion()
            print(
                f"\n\n{Bcolors.TITLE}{Bcolors.BOLD}--- Process Finished ---{Bcolors.ENDC}"
            )
            print(
                f"{Bcolors.TITLE}Full transcript saved to {self.markdown_filename}{Bcolors.ENDC}"
            )
            self.logger.info(
                f"Debate completed successfully. Transcript saved to {self.markdown_filename}"
            )
        except KeyboardInterrupt:
            print(
                f"\n\n{Bcolors.TITLE}{Bcolors.BOLD}--- Script interrupted by user ---{Bcolors.ENDC}"
            )
            print(
                f"{Bcolors.TITLE}Partial transcript saved to {self.markdown_filename}{Bcolors.ENDC}"
            )
            self.logger.warning(
                f"Debate interrupted by user. Partial transcript saved to {self.markdown_filename}"
            )
        except Exception as e:
            print(f"\n{Bcolors.BOLD}An unexpected error occurred: {e}{Bcolors.ENDC}")
            self.logger.exception(f"Unexpected error during debate: {e}")
        finally:
            self.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conduct a debate between two local LLMs with refined consensus and summary features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (optional).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Maximum number of rounds for the debate. (Default from config or 15)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="The initial topic for the debate. (Default from config or built-in default)",
    )
    parser.add_argument(
        "--min_rounds",
        type=int,
        default=None,
        help=f"The minimum number of rounds before models can conclude the debate. (Default from config or {DEFAULT_MIN_ROUNDS})",
    )
    parser.add_argument(
        "--model1_url",
        type=str,
        default=None,
        help=f"URL of the first model's completion endpoint. (Default from config or {DEFAULT_MODEL1_URL})",
    )
    parser.add_argument(
        "--model2_url",
        type=str,
        default=None,
        help=f"URL of the second model's completion endpoint. (Default from config or {DEFAULT_MODEL2_URL})",
    )
    parser.add_argument(
        "--model1_type",
        type=str,
        default=None,
        help="Manually specify the type of the first model (e.g., 'gemma', 'qwen').",
    )
    parser.add_argument(
        "--model2_type",
        type=str,
        default=None,
        help="Manually specify the type of the second model (e.g., 'gemma', 'qwen').",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (optional). If not specified, logs to stderr only.",
    )
    args = parser.parse_args()

    # Build model_urls and model_types from individual args or None
    model_urls: Optional[List[str]] = None
    if args.model1_url is not None or args.model2_url is not None:
        model_urls = [
            args.model1_url if args.model1_url is not None else DEFAULT_MODEL1_URL,
            args.model2_url if args.model2_url is not None else DEFAULT_MODEL2_URL,
        ]

    model_types: Optional[List[Optional[str]]] = None
    if args.model1_type is not None or args.model2_type is not None:
        model_types = [args.model1_type, args.model2_type]

    consensus = LLMConsensus(
        rounds=args.rounds,
        topic=args.topic,
        min_rounds=args.min_rounds,
        model_urls=model_urls,
        model_types=model_types,
        config_file=args.config,
        log_level=args.log_level,
        log_file=args.log_file,
    )
    consensus.run()
