import json 
import re
import subprocess
import tempfile
import os
import time
import shutil
import lizard
import radon.complexity as metrics
from radon.raw import analyze
from difflib import ndiff
from datetime import datetime

def extract_code_blocks(filepath):
    
    with open(filepath, encoding='utf-8') as file:
        data = json.load(file)

    user_code_blocks = [] 
    assistant_code_blocks = [] 
    user_prompts = []

    for idx in range(1, len(data), 2): #starts at idx 1 to skip the system message
        item = data[idx]
        user_content = item.get('content', '')
        assistant_content = data[idx+1].get('content', '')
        user_prompt = item.get('prompt', '')

        if re.search(r'```', assistant_content) and data[idx+1]["role"] == "assistant":
            user_code_blocks.extend(re.findall(r'```(.*?)```', replace_newlines_and_indentations(user_content), re.DOTALL))
            user_prompts.append(user_prompt)

            # work around for some weird issue, but its working
            assistant_code = re.findall(r'```(.*?)```', assistant_content, re.DOTALL)
            assistant_code = [x.replace("csharp", "") for x in assistant_code] # filtering out the "csharp" keyword
            assistant_code_new = [""]
            assistant_code_new[0] = "\n".join(assistant_code) # in case there are many code blocks in the model response, concatenate them.. 
            assistant_code = assistant_code_new
            assistant_code_blocks.extend(assistant_code)
        else:
            print("Didn't find a code block in the model response. Skipping this pair of (original, updated)")


    print("There are", len(user_code_blocks), "user code blocks and", len(assistant_code_blocks), "models code blocks.")
    if len(user_code_blocks) != len(assistant_code_blocks): print("Some of the code has not been rewritten")
    
    paired_code_blocks = list(zip(user_code_blocks, assistant_code_blocks))
        
    return paired_code_blocks, user_prompts


def calculate_metrics(code_blocks):
    og_metrics = []
    up_metrics = []
    for original_code, updated_code in code_blocks:
        # print(original_code)
        # print(updated_code)

        original_metrics = analyze_csharp_code(original_code)
        updated_metrics  = analyze_csharp_code(updated_code )
        # print("original metrics:", original_metrics)
        # print("updated metrics:", updated_metrics)

        metrics_diff = analyze_metrics_difference(original_metrics, updated_metrics)
        # print("metrics diff:", metrics_diff)
 
        original_metrics["levenshtein_distance"] = levenshtein_distance(original_code, updated_code)
        updated_metrics["levenshtein_distance"] = levenshtein_distance(original_code, updated_code)
        original_metrics["syntactic_similarity"] = syntactic_similarity(original_code, updated_code)
        updated_metrics["syntactic_similarity"] = syntactic_similarity(original_code, updated_code)
        # print("levenshtein:", levenshtein_distance(original_code, updated_code))
        # print("syntactic:", syntactic_similarity(original_code, updated_code))
        og_metrics.append(original_metrics)
        up_metrics.append(updated_metrics)
    
    return zip(og_metrics, up_metrics)


def analyze_csharp_code(code):
    """
    Args:
        code (str): The C# code snippet.
    Returns:
        dict: A dictionary containing metrics (LOC, comment lines, blank lines, characters, tokens).
    """
    lines = code.split("\n")
    loc_count = 0
    comment_count = 0
    blank_count = 0
    char_count = 0
    token_count = 0
    in_multiline_comment = False

    for line in lines:
        stripped_line = line.strip()

        if not stripped_line:
            # Blank line
            blank_count += 1
        elif stripped_line.startswith("//"):
            # Single-line comment
            comment_count += 1
        elif stripped_line.startswith("/*") and stripped_line.endswith("*/"):
            # Multi-line comment (entire line is a comment)
            comment_count += 1
        elif stripped_line.startswith("/*"):
            # Start of multi-line comment
            in_multiline_comment = True
            comment_count += 1
        elif stripped_line.endswith("*/"):
            # End of multi-line comment
            in_multiline_comment = False
            comment_count += 1
        elif in_multiline_comment:
            # Inside multi-line comment
            comment_count += 1
        else:
            # Non-empty line of code
            loc_count += 1
            char_count += len(stripped_line)
            # Token count (approximation based on whitespace-separated words)
            tokens = re.findall(r"\S+", stripped_line)
            token_count += len(tokens)

    metrics = {
        "LOC": loc_count,
        "comment": comment_count,
        "blank": blank_count,
        "chars": char_count,
        "tokens": token_count,
        "comment/LOC": -1 if loc_count == 0 else comment_count/loc_count
    }
    return metrics

def analyze_metrics_difference(original_metrics, updated_metrics):
    """
    Compares two metrics dictionaries and analyzes the differences.
    Args:
        original_metrics (dict): Metrics dictionary for the original code.
        updated_metrics (dict): Metrics dictionary for the updated code.
    Returns:
        dict: A summary of differences between the metrics.
    """
    differences = {}
    for metric, original_value in original_metrics.items():
        updated_value = updated_metrics.get(metric)
        if updated_value is not None:
            if original_value != updated_value:
                differences[metric] = {
                    "original": original_value,
                    "updated": updated_value
                }
    return differences

def PYTHON_calculate_loc_lloc_sloc_comments_blank(code):
    res = analyze(code)
    print(res, type(res))
    return res

def PYTHON_calculate_cyclomatic_complexity(function_code):
    # Compute the cyclomatic complexity
    cc = metrics.cc_visit(function_code)
    return cc[0].complexity

def levenshtein_distance(str1, str2):
    counter = {"+": 0, "-": 0}
    distance = 0
    for edit_code, *_ in ndiff(str1, str2):
        if edit_code == " ":
            distance += max(counter.values())
            counter = {"+": 0, "-": 0}
        else: 
            counter[edit_code] += 1
    distance += max(counter.values())
    return distance

def syntactic_similarity(str1, str2):
    return 1 - levenshtein_distance(str1, str2) / max(len(str1), len(str2))

def replace_newlines_and_indentations(input_string):
    # Replace newline characters with '\\n'
    input_string = input_string.replace("\\n", "\n")
    
    # Replace tab characters with '\\t'
    input_string = input_string.replace("\\t", "\t")
    
    return input_string





# --------------------------------------------------------------
# This part below will be related to cyclomatic complexity with lizard
# --------------------------------------------------------------

def create_temp_csharp_file(code, dir="tmp/"):
    # Create a temporary C# file
    temp_dir = tempfile.mkdtemp(dir=dir)
    temp_csharp_file = os.path.join(temp_dir, "temp_program.cs")

    with open(temp_csharp_file, "w", encoding='utf-8') as f:
        f.write(code)

    return temp_csharp_file


def execute_shell_command(command):
    # Execute the shell command
    try:
        output = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return None

def calculate_cyclomatic_complexity_one(csharp_code):
    """
    Calculate the cyclomatic complexity of a list of C# code snippets.

    This function takes a list of C# code snippets, creates temporary files for each snippet,
    analyzes the cyclomatic complexity using the lizard library, and returns a list of dictionaries
    containing the complexity information for each snippet. If an error occurs during analysis,
    None is appended to the result list to maintain the same length.

    Args:
        csharp_code (list of str): A list of C# code snippets as strings.

    Returns:
        list of dict or None: A list of dictionaries containing the cyclomatic complexity information
        for each code snippet. If an error occurs during analysis, None is returned for that snippet.

    Raises:
        Exception: If there is an error during the cleanup process of temporary files.
    """
    original_cyclo = []
    for original_code in csharp_code:

        temp_csharp_file_path1 = create_temp_csharp_file(original_code)

        try:
            first = lizard.analyze_file(temp_csharp_file_path1).function_list[0].__dict__
            if first:
                original_cyclo.append(first)
        except:
            original_cyclo.append(None) # this is done to keep the same length
             # TODO should actually delete files here?


        # Clean up: Delete the temporary C# file
        try:
            shutil.rmtree(temp_csharp_file_path1)
        except Exception as e:
            os.remove(temp_csharp_file_path1)
    return original_cyclo


def calculate_cyclomatic_complexity(csharp_code):
    """
    Calculate the cyclomatic complexity of original and updated C# code snippets.

    This function takes a list of tuples containing original and updated C# code snippets,
    creates temporary files for each snippet, analyzes their cyclomatic complexity using
    the lizard library, and returns a list of dictionaries containing the complexity metrics
    for both original and updated code.

    Args:
        csharp_code (list of tuples): A list where each tuple contains two strings:
            - original_code (str): The original C# code snippet.
            - updated_code (str): The updated C# code snippet.

    Returns:
        list of tuples: A list of tuples where each tuple contains two dictionaries:
            - original_cyclo (dict or None): The cyclomatic complexity metrics of the original code.
            - updated_cyclo (dict or None): The cyclomatic complexity metrics of the updated code.
            If an error occurs during analysis, None is returned for that particular code snippet.

    Raises:
        Any exceptions raised during file creation, analysis, or deletion are caught and handled
        within the function.
    """
    original_cyclo = []
    updated_cyclo = []
    for original_code, updated_code in csharp_code:

        temp_csharp_file_path1 = create_temp_csharp_file(original_code)
        temp_csharp_file_path2 = create_temp_csharp_file(updated_code)

        try:
            first = lizard.analyze_file(temp_csharp_file_path1).function_list[0].__dict__
            second = lizard.analyze_file(temp_csharp_file_path2).function_list[0].__dict__
            if first and second:
                original_cyclo.append(first)
                updated_cyclo.append(second)
        except:
            original_cyclo.append(None) # this is done to keep the same length
            updated_cyclo.append(None)
             # TODO should actually delete files here?


        # Clean up: Delete the temporary C# file
        try:
            shutil.rmtree(temp_csharp_file_path1)
            shutil.rmtree(temp_csharp_file_path2)
        except Exception as e:
            os.remove(temp_csharp_file_path1)
            os.remove(temp_csharp_file_path2)
    return zip(original_cyclo, updated_cyclo)


def combine_metrics(static_metrics, cyclo_metrics):
    for idx in range(len(static_metrics)):
        if cyclo_metrics[idx][0]:
            static_metrics[idx][0]["cyclomatic_complexity"] = cyclo_metrics[idx][0]["cyclomatic_complexity"]
        else:
            static_metrics[idx][0]["cyclomatic_complexity"] = -1

        if cyclo_metrics[idx][1]:
            static_metrics[idx][1]["cyclomatic_complexity"] = cyclo_metrics[idx][1]["cyclomatic_complexity"]
        else:
            static_metrics[idx][1]["cyclomatic_complexity"] = -1

    changed_name_metrics = []
    for idx in range(len(static_metrics)):
        entry = {}
        og_metrics, updated_metrics = static_metrics[idx]
        for key, val in og_metrics.items():
            entry["original_"+key] = val
        for key, val in updated_metrics.items():
            entry["updated_"+key] = val

        # Add comment/LOC ratio
        try:
            entry["original_comment/LOC"] = entry["original_comment"] / entry["original_LOC"]
        except Exception as e:
            entry["original_comment/LOC"] = -1
        try :
            entry["updated_comment/LOC"] = entry["updated_comment"] / entry["updated_LOC"]
        except Exception as e:
            entry["updated_comment/LOC"] = -1
        
        changed_name_metrics.append(entry)
    return changed_name_metrics


def save_to_json(paired_code_blocks, all_metrics, user_prompts, session_filepath, path = "workflow_automation/jsons/"):
    assert (len(paired_code_blocks) == len(all_metrics))

    def extract_first_lines(string):
        lines = string.split('\n')
        # 1st line is local path
        # 2nd line is start line/end line
        # 3rd line is github repo url
        # 4th line is language

        # the [3:] is a hacky way of removing the // comment 
        # the [2:] is because first line is blank, second line is the path to snippet
        return lines[1][3:], lines[2][3:].split('/'), lines[3][3:], lines[4][3:], "\n".join(lines[5:])

    def extract_func_name(string):
        lines = string.split('\n')
        return lines[0].strip()

    def extract_session_name(string):
        s = string.split('/')
        return s[-1]
    
    def choose_run_command(language):
        match language.lower():
            case "c#":
                return "dotnet build"
            case "c++":
                return "make"
            case _:
                return ""

    data = []
    
    for idx in range(len(paired_code_blocks)):
        original_code, updated_code = paired_code_blocks[idx]
        user_prompt = user_prompts[idx]
        path_to_snippet, start_end_lines, repo_url, language, original_code = extract_first_lines(original_code)
        snippet_func_name = extract_func_name(original_code)

        metrics = all_metrics[idx]

        entry = {
            "repo_url": repo_url.strip() + ".git",
            "repo_name": extract_session_name(repo_url).strip(),
            "local_path": "workflow_automation/repositories/",# + extract_session_name(repo_url).strip() + "/",   
            "path_to_snippet": path_to_snippet.strip(),
            "snippet_func_name": snippet_func_name,
            "snippet_line_start": start_end_lines[0].strip(), 
            "snippet_line_end": start_end_lines[1].strip(),
            "language": language.strip(),
            "run_command": choose_run_command(language.strip()),
            "prerequisites": "",
        } | metrics
        entry["original_code"] = original_code.rstrip()
        entry["updated_code"]  = updated_code
        entry["prompt"] = user_prompt
        
        data.append(entry)

        # break


    with open(path + extract_session_name(session_filepath), 'w+', encoding='utf-8') as json_file: 
        json.dump(data, json_file, indent=4)


def find_latest_session_filepath():
    """
    Finds the latest session file in the specified output directory based on the session name and language model.
    The function searches through the files in the 'output/' directory, looking for files that match the given 
    session name and language model. It then parses the date and time from the filenames and returns the path 
    to the file with the most recent date and time.
    Returns:
        str: The file path of the latest session file that matches the criteria.
    """
    output_path = "output/"
    session_name = "code_snippets"
    language_model = "codellama-7b-it"

    latest_date = None
    latest_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(output_path):
        # Split the file name into parts
        parts = filename.split('_')
        
        # Check if the session name and language model match
        if (parts[0] == session_name or parts[0] + "_" + parts[1] == session_name) and language_model in parts[4]:
            # Parse the date and time from the file name
            date_time_str = parts[2] + ' ' + parts[3][:-3] + ":" + parts[3][-2:]
            date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M')

            if latest_date is None or date_time > latest_date:
                latest_date = date_time
                latest_file = filename

    return output_path + latest_file


if __name__ == "__main__":

    start = time.time()
    
    session_filepath = find_latest_session_filepath()
    print(session_filepath)

    paired_code_blocks, user_prompts = extract_code_blocks(session_filepath)
    
    static_metrics = calculate_metrics(paired_code_blocks) # they're zipped together (og, updated)

    cyclo_metrics = calculate_cyclomatic_complexity(paired_code_blocks) # also zipped together (og, updated)

    all_metrics = combine_metrics(list(static_metrics), list(cyclo_metrics))
    
    save_to_json(paired_code_blocks, all_metrics, user_prompts, session_filepath)

    end = time.time()
    print("Time taken: {} seconds, {} minutes, {} hours.".format(end - start, (end-start)/60, (end-start)/3600))
