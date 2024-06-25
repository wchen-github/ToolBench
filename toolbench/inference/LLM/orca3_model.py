#!/usr/bin/env python
# coding=utf-8
import re
import time
from termcolor import colored
from typing import Optional, List
import torch
from typing import Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from toolbench.utils import process_system_message
from toolbench.model.model_adapter import get_conversation_template
from toolbench.inference.utils import SimpleChatIO, generate_stream, react_parser
from toolbench.inference.LLM.tool_llama_model import ToolLLaMA

class Orca3(ToolLLaMA):
    def __init__(
            self, 
            model_name_or_path: str, 
            template:str="tool-llama-single-round", 
            device: str="cuda", 
            cpu_offloading: bool=False, 
            max_sequence_length: int=8192
        ) -> None:
        super().__init__(model_name_or_path, template, device, cpu_offloading, max_sequence_length)
        self.process_system_message = self.process_system_message_xml
        self.parser = self.react_parser_xml

    def process_system_message_xml(self, system_message, functions):
        assert "with a function call to actually excute your step." in system_message
        # we find that following ReACT format and merging the thought node and function call node is easier for model to learn to integrate the action input json string in its prediction than learn to predict a json string directly.
        system_message = system_message.replace("with a function call to actually excute your step.", "with a function call to actually execute your step.")
        system_message = system_message + " Your output should follow this format:\n<Thought>...</Thought>\n<Action>...</Action>\n<ActionInput>...</ActionInput>\n"
        # add all the function dicts in the prompt.
        system_message = system_message + "\nSpecifically, you have access to the following APIs: " + str(functions)
        return system_message
    
    # For prediction parsing, into ReACT format
    def react_parser_xml(self, string):
        thought_pattern = r'<Thought>(.*?)</Thought>'
        thought = re.findall(thought_pattern, string, re.DOTALL)

        action_pattern = r'<Action>(.*?)</Action>'
        action = re.findall(action_pattern, string, re.DOTALL)

        action_input_pattern = r'<ActionInput>(.*?)</ActionInput>'
        action_input = re.findall(action_input_pattern, string, re.DOTALL)        
        
        if not thought or not action or not action_input:
            return [], [], []
        else:
            return thought[0], action[0], action_input[0]

if __name__ == "__main__":
    # can accept all huggingface LlamaModel family
    llm = ToolLLaMA("decapoda-research/llama-7b-hf")
    messages = [
        {'role': 'system', 'content': '''You are AutoGPT, you can use many tools(functions) to do
the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step.\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is , you can\'t go
back to the former state, if you want to restart the task, say "I give up and restart".\n2.All the thought is short, at most in 5 sentence.\nLet\'s Begin!\nTask description: Use numbers and basic arithmetic operations (+ - * /) to obtain exactly one number=24. Each
step, you are only allowed to choose two of the left numbers to obtain a new number. For example, you can combine [3,13,9,7] as 7*9 - 3*13 = 24.\nRemember:\n1.all of the number must be used , and must be used ONCE. So Only when left numbers is exact 24, you will win. So you don\'t succeed when left number = [24, 5]. You succeed when left number = [24]. \n2.all the try takes exactly 3 steps, look
at the input format'''}, 
{'role': 'user', 'content': '\nThe real task input is: [1, 2, 4, 7]\nBegin!\n'}
]
    functions = [{'name': 'play_24', 'description': '''make your current conbine with the format "x operation y = z (left: aaa) " like "1+2=3, (left: 3 5 7)", then I will tell you whether you win. This is the ONLY way
to interact with the game, and the total process of a input use 3 steps of call, each step you can only combine 2 of the left numbers, so the count of left numbers decrease from 4 to 1''','parameters':{'type': 'object', 'properties':{}}}]#, 'parameters': {'type': 'object', 'properties': {'input': {'type': 'string', 'description': 'describe what number you want to conbine, and how to conbine.'}}, 'required': ['input']}}]

    llm.change_messages(messages)
    output = llm.parse(functions=functions)
    print(output)