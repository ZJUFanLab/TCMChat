# -*- encoding: utf-8 -*-
'''
Filename         :web_demo.py
Description      :
Time             :2024/05/16 17:33:03
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''

import argparse
import os
from threading import Thread
from dataclasses import dataclass
from typing import Optional, List, Sequence

import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)

@dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system_prompt: str
    # All messages. format: list of [question, answer]
    messages: Optional[List[Sequence[str]]]
    # The roles of the speakers
    roles: Optional[Sequence[str]]
    # Conversation prompt
    prompt: str
    # Separator
    sep: str
    # Stop token, default is tokenizer.eos_token
    stop_str: Optional[str] = "</s>"

    def get_prompt(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> str:
        """
        Returns a string containing prompt without response.
        """
        return "".join(self._format_example(messages, system_prompt))

    def get_dialog(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        """
        Returns a list containing 2 * n elements where the 2k-th is a query and the (2k+1)-th is a response.
        """
        return self._format_example(messages, system_prompt)

    def _format_example(
            self,
            messages: Optional[List[Sequence[str]]] = None,
            system_prompt: Optional[str] = ""
    ) -> List[str]:
        system_prompt = system_prompt or self.system_prompt
        system_prompt = system_prompt + self.sep if system_prompt else ""  # add separator for non-empty system prompt
        messages = messages or self.messages
        convs = []
        for turn_idx, [user_query, bot_resp] in enumerate(messages):
            if turn_idx == 0:
                convs.append(system_prompt + self.prompt.format(query=user_query))
                convs.append(bot_resp)
            else:
                convs.append(self.sep + self.prompt.format(query=user_query))
                convs.append(bot_resp)
        return convs

    def append_message(self, query: str, answer: str):
        """Append a new message."""
        self.messages.append([query, answer])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="Baichuan2-7B-Chat", type=str)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--share', action='store_true', help='Share gradio')
    parser.add_argument('--port', default=8081, type=int, help='Port of gradio demo')
    args = parser.parse_args()
    print(args)

    if args.only_cpu is True:
        args.gpus = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )
    try:
        model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path, 
                                                                   trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
   
    if device == torch.device('cpu'):
        model.float()
    model.eval()
    
    prompt_template = Conversation(
        name="baichuan2",
        system_prompt="",
        messages=[],
        roles=("<reserved_106>", "<reserved_107>"),
        prompt="<reserved_106>{query}<reserved_107>",
        sep="</s>",
    )
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    def predict(message, history):
        """Generate answer from prompt with GPT and stream the output"""
        history = []
        history_messages = history + [[message, ""]]
        prompt = prompt_template.get_prompt(messages=history_messages)

        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = tokenizer(prompt).input_ids
        context_len = 2048
        max_new_tokens = 1024
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != stop_str:
                partial_message += new_token
                yield partial_message

    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="我是TCMChat智能机器人，请问有什么需要帮助的？", lines=4, scale=9),
        title="TCMChat  ",
        description="中药知识问答，本项目开源了[TCMChat](https://github.com/daiyizheng/TCMChat)中药大模型",
        theme="soft",
    ).queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()