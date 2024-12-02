import requests
import pprint
import sys
import time
import json
import os
from datetime import datetime

sys.path.append('.')

from llama_cpp import Llama

API_URL = os.getenv("API_URL")

def extract_instructions_from_markdown(markdown_file, system_prompt="You are a world-class programmer and a helpful assistant that follows requirements to the letter. Answer all queries truthfully, precisely and to the best of your ability with a single code block."):
    '''
    This function is to be used when the markdown file (the session) is 
    a just series of ## User instructions, without the model replies.
    '''
    final_content = [{
                "content": system_prompt,
                "role": "system"
            }]

    with open(markdown_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    role = None
    content = []

    for line in lines:
        if line.startswith("## User"):
            if content and role:
                final_content.append({"role": role, "content": " ".join(content), "prompt": content[-2]})
            role = "user"
            content = []
        else:
            content.append(line)#.strip())

    if content:
        final_content.append({"role": role, "content": " ".join(content),  "prompt": content[-1]})

    return final_content


def send_completion_request(chat):
    headers = {"Content-Type": "application/json"}
    payload = { "messages": chat }

    # print("Payload:", payload)

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            print("Request successful.")
            # print(response.json())
            return response.json()
        else:
            print(f"Request failed with status code {response.status_code}.")
    except requests.RequestException as e:
        print(f"Error sending request: {e}")

def extract_model_response(response):
    return [{"role": "assistant", "content": response["choices"][0]["message"]["content"].strip()}]


def replay_session(markdown_file_path):
    exctracted_instructions = extract_instructions_from_markdown(markdown_file_path)

    if len(exctracted_instructions) < 2: sys.exit("Markdown isntructions doesn't contain instructions.")
    if len(exctracted_instructions) > 5: return replay_session_slowly(markdown_file_path)

    current_chat = exctracted_instructions[:2]
    response = send_completion_request(current_chat)
    current_chat += extract_model_response(response)

    for index in range(2, len(exctracted_instructions)):
        current_chat += [exctracted_instructions[index]]
        response = send_completion_request(current_chat)
        current_chat += extract_model_response(response)

    print("Done with the conversation")
    # print(current_chat)
    return current_chat

def replay_session_slowly(markdown_file_path):
    """
    the difference is that this function will not continue the chat. it will only send the system message
    and one message at a time, until the end. this is done for when context window is exceeded.
    """
    exctracted_instructions = extract_instructions_from_markdown(markdown_file_path)

    system_message = exctracted_instructions[0]
    final_chat = [exctracted_instructions[0]] + [replace_newlines_and_indentations(exctracted_instructions[1])]

    current_chat = exctracted_instructions[:2]
    response = send_completion_request(current_chat)
    final_chat += extract_model_response(response)

    for index in range(2, len(exctracted_instructions)):
        current_chat = [system_message, exctracted_instructions[index]]
        response = send_completion_request(current_chat)
        final_chat += [replace_newlines_and_indentations(exctracted_instructions[index])]
        final_chat += extract_model_response(response)

    print("Done with the conversation")
    # print(current_chat)
    return final_chat


def beautify_chat_session(chat):
    result = []
    for message in chat:
        match message["role"].lower():
            case "system":
                pass
            case "assistant":
                result.append("Model: " + message["content"])
            case "user":
                result.append("User: " + message["content"])
    pprint.pprint(result)


def save_chat_session(chat, session_name, model_name = "_codellama-7b-it", output_directory = "output/"):
    filename = session_name.split("/")[-1][:-3]
    curr_time = datetime.now().strftime('_%Y-%m-%d_%H:%M')
    with open(output_directory + filename + curr_time + model_name + ".json", 'w', encoding='utf-8') as out:
        json.dump(chat, out)

def replace_newlines_and_indentations(input_string):
    # Replace newline characters with '\n'
    input_string["content"] = input_string["content"].replace("\n", "\\n")
    
    # Replace tab characters with '\t'
    input_string["content"] = input_string["content"].replace("\t", "\\t")
    
    return input_string

def use_llama_cpp_bindings(question, model_path, chat_format="llama-2"):
    '''
    This function uses llama.cpp python bindings library to serve the model, answer the query and close the chat.
    '''
    llm = Llama(model_path=model_path, chat_format=chat_format)
    print(llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that outputs in JSON.",
            },
            {"role": "user", "content": question},
        ],
        response_format={
            "type": "json_object",
        },
        temperature=0.2,
    ))


if __name__ == "__main__":
    # use_llama_cpp_bindings("Who won the world series in 2020")
    # Example usage:
    markdown_file_path = "sessions/code_snippets.md"
    extracted_discussion = extract_instructions_from_markdown(markdown_file_path)

    start = time.time()

    chat = replay_session(markdown_file_path)
    beautify_chat_session(chat)
    save_chat_session(chat, markdown_file_path)

    end = time.time()

    print("Time taken: {} seconds, {} minutes, {} hours.".format(end - start, (end-start)/60, (end-start)/3600))
    