#!/usr/bin/env python3
"""
LLM Consensus Script - V3.7 - Minimum Rounds & Refined Instructions

This version prevents premature endings by enforcing a minimum round count
before consensus can be proposed and by providing much clearer instructions
to the models on how to conclude the debate.
"""

import requests
import json
import time
import argparse
import textwrap
from typing import List, Dict

# --- Configuration ---
CONSENSUS_PHRASE = "[DEBATE_COMPLETE]"
# Set a default minimum number of rounds (one turn per model) before consensus is allowed
DEFAULT_MIN_ROUNDS = 3 # This means at least 6 total turns must happen

class Bcolors:
    MODEL1 = '\033[94m'
    MODEL2 = '\033[92m'
    HEADER = '\033[95m'
    TITLE = '\033[93m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

class LLMConsensus:
    def __init__(self, rounds: int, topic: str, min_rounds: int):
        self.models = {
            "model1": {"name": "Model 1 (Gemma)", "url": "http://localhost:5300/completion", "template": "gemma", "stop_tokens": ["<end_of_turn>", "<start_of_turn>"], "color": Bcolors.MODEL1},
            "model2": {"name": "Model 2 (Qwen)", "url": "http://192.168.2.55:5300/completion", "template": "qwen", "stop_tokens": ["<|im_end|>"], "color": Bcolors.MODEL2}
        }
        self.max_rounds = rounds
        self.initial_topic = topic
        self.min_rounds_before_consensus = min_rounds
        self.conversation_history: List[Dict] = []
        self.session = requests.Session()
        self.markdown_filename = f"debate_{time.strftime('%Y-%m-%d_%H-%M-%S')}.md"
        self.end_proposed_by: str = None

    def format_prompt(self, target_model_key: str, is_summary_prompt: bool = False) -> str:
        """Builds a prompt using the correct chat template with refined instructions."""
        template_style = self.models[target_model_key]["template"]
        history = self.conversation_history.copy()

        # --- REFINED INSTRUCTIONS ---
        system_instruction = (
            "You are an expert AI debater. Your goal is to engage in a thorough and critical discussion. A few rules:\n"
            "- Justify your claims with evidence and logic.\n"
            "- Directly address the other model's points in your rebuttals.\n"
            "- The debate should continue for several rounds to explore the topic fully.\n"
            f"- To end the debate, you must propose it. Only do this if all arguments have been exhausted and the conversation is becoming repetitive. Propose ending by including the exact phrase: {CONSENSUS_PHRASE}\n"
            "- IMPORTANT: Do NOT use this phrase just to end your turn. It is a proposal to end the entire debate."
        )

        if is_summary_prompt:
            summary_instruction = "The following is a complete transcript of a debate. Your task is to act as a neutral moderator and write a final, conclusive summary including key arguments, conclusions, and next steps."
            history.append({"role": "user", "content": summary_instruction})
        
        elif self.end_proposed_by:
            proposer_name = self.models[self.end_proposed_by]['name']
            confirmation_instruction = (
                f"{proposer_name} has proposed concluding the debate. Please provide your final thoughts. If you agree that the "
                "discussion is complete and no new points can be made, include the phrase [DEBATE_COMPLETE] in your response. "
                "Otherwise, continue the debate by presenting a new argument."
            )
            history.append({"role": "user", "content": confirmation_instruction})

        if template_style == "gemma":
            prompt = ""
            if history:
                first_turn_content = f"{system_instruction}\n\nDEBATE TOPIC: {history[0]['content']}"
                prompt += f"<start_of_turn>user\n{first_turn_content}<end_of_turn>\n"
                for i, entry in enumerate(history[1:], 1):
                    role = "model" if entry.get('is_model_response') else "user"
                    prompt += f"<start_of_turn>{role}\n{entry['content']}<end_of_turn>\n"
            prompt += "<start_of_turn>model\n"
            return prompt
            
        elif template_style == "qwen":
            prompt = f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
            for entry in history:
                prompt += f"<|im_start|>{entry['role']}\n{entry['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt
            
        return ""

    def send_request(self, model_key: str, prompt: str, stop_override: List = None) -> str:
        model_config = self.models[model_key]
        try:
            stop_tokens = stop_override if stop_override is not None else model_config["stop_tokens"]
            data = {"prompt": prompt, "temperature": 0.7, "n_predict": 2048, "stop": stop_tokens}
            headers = {"Content-Type": "application/json"}
            response = self.session.post(model_config["url"], data=json.dumps(data), headers=headers, timeout=300)
            response.raise_for_status()
            result = response.json()
            content = result.get('content', "Error: 'content' key not found.").strip()
            return content
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to the model."

    def print_and_write_response(self, md_file, turn: int, model_config: Dict, response: str, exec_time: float):
        """Helper to print to shell and write to markdown file."""
        # Print to shell
        header_text = f" Turn {turn + 1}: {model_config['name']}'s Response "
        print(f"{Bcolors.HEADER}{Bcolors.BOLD}{'=' * 10}{header_text}{'=' * (90 - len(header_text))}{Bcolors.ENDC}")
        print(f"{model_config['color']}{textwrap.fill(response, width=100)}{Bcolors.ENDC}")
        print(f"{Bcolors.HEADER}{Bcolors.BOLD}{'=' * 90} [Response Time: {exec_time:.2f}s]{Bcolors.ENDC}\n")
        # Write to file
        md_file.write(f"## Turn {turn + 1}: {model_config['name']}\n\n")
        for line in response.split('\n'):
            md_file.write(f"> {line}\n")
        md_file.write(f"\n_Response Time: {exec_time:.2f}s_\n\n---\n\n")

    def conduct_discussion(self) -> None:
        """Conducts a turn-by-turn discussion with refined consensus logic."""
        print(f"{Bcolors.TITLE}{Bcolors.BOLD}{'='*60}\n      STARTING LLM CONSENSUS DISCUSSION (V3.7 - Refined Consensus)\n{'='*60}{Bcolors.ENDC}")
        print(f"{Bcolors.TITLE}Saving transcript to: {self.markdown_filename}{Bcolors.ENDC}")
        print(f"{Bcolors.HEADER}DEBATE TOPIC:{Bcolors.ENDC} \"{self.initial_topic}\"")

        with open(self.markdown_filename, 'w', encoding='utf-8') as md_file:
            md_file.write(f"# LLM Debate Transcript\n\n**Topic:** {self.initial_topic}\n\n---\n\n")

            self.conversation_history.append({"role": "user", "content": self.initial_topic})
            current_speaker_key = "model1"
            
            for turn in range(self.max_rounds * 2):
                model_config = self.models[current_speaker_key]
                prompt = self.format_prompt(current_speaker_key)
                
                start_time = time.time()
                response = self.send_request(current_speaker_key, prompt)
                exec_time = time.time() - start_time

                debate_concluded = False
                clean_response = response.replace(CONSENSUS_PHRASE, "").strip()
                
                # Check if consensus phrase is used AND we are past the minimum number of rounds
                if CONSENSUS_PHRASE in response and (turn + 1) > (self.min_rounds_before_consensus * 2):
                    if not self.end_proposed_by:
                        self.end_proposed_by = current_speaker_key
                        print(f"\n{Bcolors.TITLE}{Bcolors.BOLD}--- {model_config['name']} has proposed concluding the debate. Awaiting confirmation... ---{Bcolors.ENDC}")
                        md_file.write(f"**__{model_config['name']} has proposed concluding the debate.__**\n\n")
                    else:
                        print(f"\n{Bcolors.TITLE}{Bcolors.BOLD}--- {model_config['name']} agrees. Both models have concluded the debate. ---{Bcolors.ENDC}")
                        md_file.write(f"**__{model_config['name']} agrees. The debate has concluded by consensus.__**\n\n")
                        debate_concluded = True
                elif self.end_proposed_by:
                    print(f"\n{Bcolors.TITLE}{Bcolors.BOLD}--- {model_config['name']} continues the debate, rejecting the proposal to end. ---{Bcolors.ENDC}")
                    md_file.write(f"**__{model_config['name']} continues the debate.__**\n\n")
                    self.end_proposed_by = None

                self.print_and_write_response(md_file, turn, model_config, clean_response, exec_time)
                
                self.conversation_history.append({"role": 'assistant', "content": clean_response, "is_model_response": True})
                
                if debate_concluded:
                    break

                current_speaker_key = "model2" if current_speaker_key == "model1" else "model1"
            
            self.generate_final_summary(md_file)

    def generate_final_summary(self, md_file):
        """Tasks a model to write a final summary of the debate."""
        print(f"\n{Bcolors.TITLE}{Bcolors.BOLD}{'='*30} Generating Final Summary {'='*30}{Bcolors.ENDC}")
        summarizer_key = "model1"
        summary_prompt = self.format_prompt(summarizer_key, is_summary_prompt=True)
        summary_response = self.send_request(summarizer_key, summary_prompt, stop_override=[])
        
        final_summary_header = f"--- Final Summary (written by {self.models[summarizer_key]['name']}) ---"
        print(f"{Bcolors.HEADER}{Bcolors.BOLD}{final_summary_header}{Bcolors.ENDC}")
        print(f"{Bcolors.MODEL1}{textwrap.fill(summary_response, width=100)}{Bcolors.ENDC}")
        
        md_file.write("## Final Summary & Conclusion\n\n")
        md_file.write(f"_This summary was generated by {self.models[summarizer_key]['name']}._\n\n")
        md_file.write(summary_response)

    def run(self):
        """Main execution method, fully restored."""
        try:
            self.conduct_discussion()
            print(f"\n\n{Bcolors.TITLE}{Bcolors.BOLD}--- Process Finished ---{Bcolors.ENDC}")
            print(f"{Bcolors.TITLE}Full transcript saved to {self.markdown_filename}{Bcolors.ENDC}")
        except KeyboardInterrupt:
            print(f"\n\n{Bcolors.TITLE}{Bcolors.BOLD}--- Script interrupted by user ---{Bcolors.ENDC}")
            print(f"{Bcolors.TITLE}Partial transcript saved to {self.markdown_filename}{Bcolors.ENDC}")
        except Exception as e:
            print(f"\n{Bcolors.BOLD}An unexpected error occurred: {e}{Bcolors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conduct a debate between two local LLMs with refined consensus and summary features.")
    parser.add_argument("--rounds", type=int, default=15, help="Maximum number of rounds for the debate.")
    parser.add_argument("--topic", type=str, default="Let's debate which of us is the 'smarter' AI.", help="The initial topic for the debate.")
    parser.add_argument(
        "--min_rounds",
        type=int,
        default=DEFAULT_MIN_ROUNDS,
        help=f"The minimum number of rounds before models can conclude the debate. (Default: {DEFAULT_MIN_ROUNDS})"
    )
    args = parser.parse_args()
    
    consensus = LLMConsensus(rounds=args.rounds, topic=args.topic, min_rounds=args.min_rounds)
    consensus.run()