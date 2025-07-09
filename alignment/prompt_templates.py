import os
from typing import Union

TEMPLATES_PATH = os.path.join("alignment", "prompts")

PROMPT_FORMATS = {
    f.split(".")[0]: open(os.path.join(TEMPLATES_PATH, f), 'r').read() 
    for f in os.listdir(TEMPLATES_PATH)
}

def format_prompt(prompts: Union[list[str], str], format: str | None = "r1_zero")-> str:
    if isinstance(prompts, str):
        prompts = [prompts]
    
    if format is None:
        return prompts

    assert format in PROMPT_FORMATS.keys()
    formated_prompts = []
    template = PROMPT_FORMATS[format]

    for p in prompts:
        formated_prompts.append(template.format(question=p))

    return formated_prompts

if __name__ == "__main__":
    prompts = ["1 + 1 = ?"]
    print(format_prompt(prompts))