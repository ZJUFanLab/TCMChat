import os, sys
import argparse
import time

import pandas as pd
from loguru import logger

sys.path.insert(0, "./")

from evaluation.choices_evaluate.chatgpt import ChatGPT_Evaluator
from evaluation.choices_evaluate.gemini import Gemini_Evaluator
from evaluation.choices_evaluate.baichuan import Baichuan_Evaluator
from evaluation.choices_evaluate.llama_hf import LLaMA_HuggingFace_Evaluator
from evaluation.choices_evaluate.general import General_Evaluator


def main(args):
    logger.info(args)
    model_name = args.model_name.lower().strip()
    if  model_name in ["gpt-3.5-turbo", "gpt-4"]:
        evaluator=ChatGPT_Evaluator(
            choices=args.choices,
            k=args.ntrain,
            api_secret_key=args.api_secret_key,
            base_url=args.base_url,
            model_name=args.model_name,
            model_path_or_name = args.model_path_or_name
        )
    elif model_name in ["gemini-pro"]:
        evaluator=Gemini_Evaluator(
            choices=args.choices,
            k=args.ntrain,
            api_secret_key=args.api_secret_key,
            base_url=args.base_url,
            model_name=args.model_name,
            model_path_or_name = args.model_path_or_name
        )
    elif args.model_name.lower() in ["baichuan", 
                                     "baichuan2", 
                                     "baichuan2-7b-chat",
                                     "baichuan2-13b-chat",
                                     "cmlm-zhongjing", 
                                     "huatuogpt",
                                     "tcmchat"]:
        evaluator=Baichuan_Evaluator(
            choices=args.choices,
            k=args.ntrain,
            model_name=args.model_name,
            model_path_or_name = args.model_path_or_name
        )
    elif args.model_name.lower().strip() in ["bentsao", "bentsao-literature", "bentsao-med"]:
        evaluator= LLaMA_HuggingFace_Evaluator(
            choices=args.choices,
            model_name=args.model_name,
            model_path_or_name = args.model_path_or_name,
            k=args.ntrain)
    elif model_name in ["bianque-2", "bianque"]:
        evaluator=General_Evaluator(
            choices=args.choices,
            k=args.ntrain,
            model_name=args.model_name,
            model_path_or_name = args.model_path_or_name,
            is_half = args.is_half
        )
    else:
        raise ValueError("Unknown model name")

 
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
    save_result_dir=os.path.join(log_dir, f"{args.model_name}_{run_date}")
    os.mkdir(save_result_dir)

    val_file_path=args.val_file_path
    val_df=pd.read_csv(val_file_path)
    choices_type_name = "单项选择题"
    if args.choices_type=="single":
        choices_type_name = "单项选择题"
    elif args.choices_type == "multiple":
        choices_type_name = "多项选择题"
    else:
        raise ValueError
    if args.few_shot:
        dev_file_path=args.dev_file_path
        dev_df=pd.read_csv(dev_file_path)
        correct_ratio = evaluator.eval_subject(args.subject_zh, 
                                               val_df, 
                                               choices_type_name,
                                               dev_df, 
                                               few_shot=args.few_shot,
                                               save_result_dir=save_result_dir,
                                               cot=args.cot)
    else:
        correct_ratio = evaluator.eval_subject(args.subject_zh, 
                                               val_df, 
                                               choices_type_name,
                                               few_shot=args.few_shot,
                                               save_result_dir=save_result_dir)
    logger.info(f"Acc:{correct_ratio}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--api_secret_key", type=str, default="xxx")
    parser.add_argument("--base_url", type=str, default="xxx")
    parser.add_argument("--minimax_group_id", type=str, default="xxx")
    parser.add_argument("--minimax_key", type=str, default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_path_or_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dev_file_path", type=str, required=True)
    parser.add_argument("--val_file_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--subject_zh","-sz",type=str, default="unk")
    parser.add_argument("--choices", "-c", nargs="+", type=int, default=["A", "B", "C", "D", "E"])
    parser.add_argument("--choices_type", type=str, default="single")
    parser.add_argument("--cuda_device", type=str)    
    parser.add_argument("--is_half", action="store_true", default=False)  
    args = parser.parse_args()
    main(args)