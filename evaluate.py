from alignment.prompt_templates import format_prompt
from argument import parse_arg, build_hf_config
from alignment.model_utils import get_vllm_models_with_sampling_param
from alignment.dataset import load_dataset
from math_verify import parse
from alignment.drgrpo_grader import r1_zero_reward_fn
from typing import Callable
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import pickle
import os

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: dict[str, list[str]],
    eval_sampling_params: SamplingParams,
    save_path: None |str = None,
    lora_request: None | LoRARequest = None
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    questions = prompts["questions"]
    ground_truth = prompts["answers"]
    outputs = vllm_model.generate(questions, eval_sampling_params, lora_request=lora_request)

    tbe_saved = []
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        data = {
            "prompt": output.prompt,
            "response": response,
            "reward": reward_fn(output.outputs[0].text, ground_truth[idx])
        }
        tbe_saved.append(data)

    if save_path is not None:
        try:
            os.makedirs(args.log_dir, exist_ok=True)
            save_path = os.path.join(args.log_dir, f"{args.model}_{args.dataset}_eval.pkl")
            with open(save_path, 'wb') as file:
                # 3. Use pickle.dump() to save the object to the file
                pickle.dump(tbe_saved, file)
            print(f"Data successfully saved to {save_path}")

        except IOError as e:
            print(f"Error: Could not save data to {save_path}. Reason: {e}")

    num_prompts = len(questions)
    format_accuracy = sum([r["reward"]["format_reward"] for r in tbe_saved]) / num_prompts
    answer_accuracy = sum([r["reward"]["answer_reward"] for r in tbe_saved]) / num_prompts
    
    print(f"format accuracy: {format_accuracy}, answer accuracy: {answer_accuracy}")

def evaluate(args):
    dataset = load_dataset(args)
    llm, sampling_params = get_vllm_models_with_sampling_param(args)
    
    # for data in dataset["test"]:
    prompts = {
        "questions": format_prompt([d["question"] for d in dataset["test"]]),
        "answers": [parse(d["answer"])[-1] for d in dataset["test"]]
    }
    save_path = os.path.join(".", "eval_data", "zero_shot_eval.pkl")
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, sampling_params, save_path=save_path)
    
if __name__ == "__main__":

    args = parse_arg()
    build_hf_config(args)
    evaluate(args)