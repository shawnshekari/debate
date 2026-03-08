# LLM Debate Script (Surrender-Based)

An interactive CLI tool that orchestrates automated debates between two local LLM instances, managing turn-taking, prompt formatting for different model architectures, and surrender-based conclusion.

## Features

- **Multi-model support**: Auto-detects and supports Gemma, Qwen, Mistral, and GPT-OSS architectures
- **Surrender-based ending**: First model to surrender loses, winner writes triumphant summary
- **Colored terminal output**: Visual distinction between models with color-coded responses
- **Markdown transcripts**: Saves complete debate history with timestamps
- **User intervention**: Models can request user input during the debate
- **Automatic summarization**: Winner explains why they won in the final summary

## Installation

```bash
# Clone or navigate to the repository
cd /path/to/debate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (for testing)
pip install pytest mypy types-requests
```

## Quick Start

### Basic Usage

```bash
# Run a debate with default settings (both models on localhost)
python debate.py --topic "Is artificial intelligence beneficial to humanity?"

# Run with custom models
python debate.py \
    --topic "Python vs Rust for systems programming" \
    --model1_url http://localhost:8080/completion \
    --model2_url http://192.168.1.100:8080/completion
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--topic` | The debate topic | "Let's debate which of us is the 'smarter' AI." |
| `--rounds` | Maximum number of debate rounds | 15 |
| `--min_rounds` | Minimum rounds before consensus can be proposed | 3 |
| `--model1_url` | URL of first model's completion endpoint | http://localhost:5300/completion |
| `--model2_url` | URL of second model's completion endpoint | http://192.168.2.55:5300/completion |
| `--model1_type` | Manual model type override (gemma, qwen, mistral, gpt-oss) | None (auto-detect) |
| `--model2_type` | Manual model type override (gemma, qwen, mistral, gpt-oss) | None (auto-detect) |
| `--config` | Path to JSON configuration file | None |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `--log-file` | Path to log file (optional) | None |

### Example Commands

```bash
# Debate with specific model types
python debate.py \
    --topic "Should AI be regulated?" \
    --model1_url http://localhost:8080 \
    --model1_type gemma \
    --model2_url http://localhost:8081 \
    --model2_type qwen

# Limited debate (5 rounds, must debate at least 2)
python debate.py \
    --topic "Remote work vs office work" \
    --rounds 5 \
    --min_rounds 2

# Using a config file
python debate.py --config my_config.json
```

## Configuration

### Config File Format

Create a JSON file (e.g., `config.json`):

```json
{
  "rounds": 10,
  "topic": "Your debate topic here",
  "min_rounds": 3,
  "model1_url": "http://localhost:8080/completion",
  "model2_url": "http://192.168.1.100:8080/completion",
  "model1_type": "gemma",
  "model2_type": "qwen",
  "log_level": "INFO",
  "log_file": "debate.log"
}
```

Then run:

```bash
python debate.py --config config.json
```

### Config Validation

The script validates configuration files and warns about issues:

- **Type checking**: Ensures values are the correct type (int, string, etc.)
- **Range validation**: Rounds must be positive and ≤ 100
- **Relationship validation**: min_rounds cannot exceed rounds
- **URL validation**: Model URLs must be valid HTTP/HTTPS URLs
- **Model type validation**: Model types must exist in model_templates.json
- **Log level validation**: Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL

Example warnings:
```
Warning: Config 'rounds' must be positive, got -5
Warning: Config 'model1_type' 'nonexistent' not found in model_templates.json
Warning: Config 'min_rounds' (10) cannot exceed 'rounds' (5)
```

### Model Templates

The script uses `model_templates.json` for chat template configurations:

```json
{
  "gemma": {
    "name": "Gemma",
    "template": "gemma",
    "stop_tokens": ["<end_of_turn>"]
  },
  "qwen": {
    "name": "Qwen",
    "template": "qwen",
    "stop_tokens": ["<|im_end|>"]
  },
  "mistral": {
    "name": "Mistral",
    "template": "mistral",
    "stop_tokens": ["</s>"]
  },
  "gpt-oss": {
    "name": "GPT-OSS",
    "template": "gpt-oss",
    "stop_tokens": ["<|im_end|>"]
  }
}
```

## How It Works

### Debate Flow

1. **Initialization**: Models are detected and configured
2. **Topic Introduction**: The debate topic is presented to Model 1
3. **Turn-taking**: Models alternate responses, building conversation context
4. **Surrender Detection**: 
   - After `min_rounds`, a model can surrender by including `[I CONCEED]`
   - The debate ends immediately, the other model wins
5. **Summary**: The winner (or Model 1 if no surrender) generates a triumphant summary

### Surrender Mechanism

```
Model 1: [Makes argument] ... I concede that Model 2 has made better points. [I CONCEED]
-> Debate ends immediately
-> Model 2 wins and writes summary explaining why they won
```

If neither model surrenders and max rounds are reached, Model 1 writes a summary.
If Model 2 presents new argument -> Debate continues
```

### Color Coding

| Color | Meaning |
|-------|---------|
| Blue | Model 1 responses |
| Green | Model 2 responses |
| Purple | Headers and separators |
| Yellow | Titles and important messages |

### Progress Indicator

During the debate, a progress bar shows the current status:

```
[Round 1/15] [████████████████████░░░░░░░░░░░░░░░░░░░░] 6.7% (Turn 1/30)
[Round 1/15] [████████████████████████████████████████] 100.0% (Turn 30/30)
```

Shows:
- Current round vs maximum rounds
- Visual progress bar
- Percentage complete
- Current turn vs maximum turns

## Output

### Terminal Output

Colored, formatted debate transcript displayed in real-time:

```
============================================================
      STARTING LLM DEBATE (V4.0 - Surrender-Based)
============================================================
Saving transcript to: debate_2024-01-15_14-30-00.md
DEBATE TOPIC: "Is artificial intelligence beneficial to humanity?"

========== Turn 1: Model 1 (gemma)'s Response ========== [Response Time: 2.34s]

[Model 1's response text...]

========== Turn 2: Model 2 (qwen)'s Response ========== [Response Time: 3.12s]

[Model 2's response text...]
```

### Markdown Transcript

A file named `debate_YYYY-MM-DD_HH-MM-SS.md` is created containing:
- Full debate topic
- All turns with model names and response times
- User interventions (if any)
- Final summary

## Development

### Running Tests

```bash
# Run all tests
pytest test_debate.py -v

# Run specific test class
pytest test_debate.py::TestValidateUrl -v

# Run with coverage
pytest --cov=debate test_debate.py
```

### Type Checking

```bash
# Run mypy
mypy debate.py
```

### Architecture

```
debate.py
├── LLMConsensus (main class)
│   ├── __init__          - Initialize debate orchestrator
│   ├── load_config       - Load configuration from JSON
│   ├── validate_inputs   - Validate all input parameters
│   ├── _sanitize_input   - Sanitize user input (security)
│   ├── load_model_templates - Load chat templates
│   ├── initialize_models - Detect and configure models
│   ├── identify_model    - Auto-detect model type
│   ├── format_prompt     - Build prompts with correct template
│   ├── send_request      - Send requests with retry logic
│   ├── conduct_discussion - Main debate loop
│   ├── generate_final_summary - Generate summary
│   └── run              - Main execution method
```

## Model Server Requirements

The script works with llama.cpp server or compatible APIs that:

1. Accept POST requests with JSON body:
   ```json
   {
     "prompt": "...",
     "temperature": 0.7,
     "n_predict": 2048,
     "stop": ["..."]
   }
   ```

2. Return JSON response:
   ```json
   {
     "content": "Model response text"
   }
   ```

## Logging

The script supports configurable logging for debugging and auditing purposes.

### Logging Levels

| Level | Description |
|-------|-------------|
| DEBUG | Detailed information for debugging (model identification, prompt/response lengths) |
| INFO | General operational information (debate start/end, summary generation) |
| WARNING | Warning conditions (retries, fallbacks) |
| ERROR | Error conditions (connection failures, timeouts) |
| CRITICAL | Critical errors (application-level failures) |

### Usage

```bash
# Log to stderr only (default: INFO level)
python debate.py --topic "AI safety"

# Enable debug logging
python debate.py --log-level DEBUG --topic "AI safety"

# Log to both stderr and file
python debate.py --log-level DEBUG --log-file debate.log --topic "AI safety"

# Only show warnings and errors
python debate.py --log-level WARNING --topic "AI safety"
```

### Log File Format

Logs are written with timestamps:

```
2024-01-15 14:30:00 - INFO - Initializing models
2024-01-15 14:30:01 - INFO - Model 1: Using template 'gemma' (identified as gemma)
2024-01-15 14:30:02 - DEBUG - Sending request to model1 (prompt length: 1234)
2024-01-15 14:30:05 - DEBUG - Received response from model1 (length: 567)
```

## Troubleshooting

### Connection Errors

```
Error: Could not connect to Model 1 (gemma)
```

- Verify the model server is running
- Check the URL is correct (including port)
- Ensure firewall allows connections

### Model Not Detected

```
Could not identify model type. Using template: 'default'
```

- Use `--model1_type` or `--model2_type` to specify manually
- Check that model server responds to identification prompts

### Premature Debate End

If debates end too quickly:

```bash
python debate.py --min_rounds 5  # Require at least 5 rounds
```

## License

MIT License

## Contributing

Contributions welcome! Please ensure:
1. Tests pass: `pytest test_debate.py -v`
2. Type checking passes: `mypy debate.py`
3. Follow existing code style
