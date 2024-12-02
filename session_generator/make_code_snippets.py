import os

PROMPTS = []
SNIPPETS = []

def extract_snippets():
    path = "session_generator/snippets/"
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                SNIPPETS.append(content)

def extract_prompts():
    path = "session_generator/prompt_formulation.md"

    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("- "):
                PROMPTS.append(line[2:].rstrip())

def add_text_to_result(text):
    path = "sessions/code_snippets.md"

    with open(path, "a", encoding='utf-8') as file:
        file.write(text)


if __name__ == "__main__":
    extract_snippets()
    extract_prompts()
    
    for snippet in SNIPPETS:
        for prompt in PROMPTS:
            text = """\n## User 
```\n{}\n```
{}\n""".format(snippet, prompt)
            add_text_to_result(text)