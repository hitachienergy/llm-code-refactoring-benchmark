import json
import os
import git 
import time
import subprocess 
import tempfile
import sys
import shutil
from datetime import datetime


class Automator:
    def __init__(self, path_to_jsons) -> None:
        self.path_to_db = path_to_jsons

    def read_json(self, path_to_json):
        try:
            with open(path_to_json, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                return data
        except FileNotFoundError:
            print(f"File not found: {path_to_json}")
            sys.exit()

    def write_json(self, data, path_to_json):
        try:
            with open(path_to_json, "w+", encoding='utf-8') as file:
                json.dump(data, file, indent=4)
        except:
            print(f"Couldn't write json {path_to_json}")
            sys.exit()

    def clone_repo(self, repo_url, temp_dir):
        def create_dirs(dir_path):
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
        try:
            create_dirs(temp_dir)
            repo = git.Repo.clone_from(repo_url, temp_dir)
            return repo
        except Exception as e:
            print("Couldn't clone repo. ", e)

    def replace_snippet(self, target_repo_path, snippet_info):
        try:
            with open(target_repo_path + "/" + snippet_info["path_to_snippet"], 'r', encoding='utf-8') as file:
                lines = file.readlines()

            if snippet_info["snippet_line_start"] and snippet_info["snippet_line_end"]:
                updated_code_lines = snippet_info["updated_code"].splitlines()
                lines[int(snippet_info["snippet_line_start"]) : int(snippet_info["snippet_line_end"])+1] = updated_code_lines 
            
            with open(target_repo_path + "/" + snippet_info["path_to_snippet"], 'w', encoding='utf-8') as file:
                file.writelines(lines)

        except Exception as e:
            print("Failed to replace snippet:", e)
            
    def compile_and_test(self, target_repo_path, compile_command, test_command):
        os.chdir(target_repo_path)
        print(target_repo_path)
        try:
            result = subprocess.run(compile_command, shell=True, check=True, text=True, capture_output=True, timeout=15)
            if "Build succeeded" in result.stdout:
                build_success = True
        except Exception as e:
            print("Failed to build.", e)
            build_success = False
        
        
        test_success = None
        #subprocess.run(test_command, shell=True)
        return build_success, test_success

    def update_build_test_metrics(self, build_success, test_success, snippet_info):
        snippet_info["build_success"] = 1 if build_success else 0
        snippet_info["test_success"] = test_success if test_success else ""

    def run(self):
        json_elements = self.read_json(self.path_to_db)
        ran_into_exception = False
        cwd = os.getcwd() + "/"

        builds_succeeded = builds_failed = 0

        for snippet_info in json_elements:
            os.chdir(cwd)

            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            os.makedirs(temp_dir, exist_ok=True)

            try: 
                # This deals with downloading/cloning the repo.
                # If source repository has already been downloaded
                print("******************************************************")
                local_path = snippet_info["local_path"] + snippet_info["repo_name"] + "_repo/" + snippet_info["repo_name"]
                if os.path.isdir(local_path):
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    shutil.copytree(local_path, temp_dir, dirs_exist_ok=True)                    
                    print("Copied local repo to temp dir.")              
                else:
                    # Clone source repository
                    source_repo = self.clone_repo(snippet_info["repo_url"], local_path + "/")
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    shutil.copytree(local_path, temp_dir, dirs_exist_ok=True)  
                    print("Downloaded repo locally and cloned it to temp dir.")

                # Replace snippet in target repository
                self.replace_snippet(temp_dir, snippet_info)
                print("Replaced")

                # Compile and test
                build_success, test_success = self.compile_and_test(temp_dir, snippet_info["run_command"], snippet_info["run_command"])
                if build_success: builds_succeeded += 1
                else: builds_failed += 1
                print("Built and tested")
                
                self.update_build_test_metrics(build_success, test_success, snippet_info)
                
                # Delete temp repo
                shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as e:
                print("Exception.", e)
                shutil.rmtree(temp_dir)
                ran_into_exception = True

        print("******************************************************")
        print("Builds succeeded: ", builds_succeeded)
        print("Builds failed: ", builds_failed)

        # TODO for now im handling each exception by itself
        # if not ran_into_exception:
        os.chdir(cwd)
        self.write_json(json_elements, self.path_to_db)
        print("Updated metrics and saved to json")
            

def find_latest_jsons(output_path = "workflow_automation/jsons/"):
    session_name = "code_snippets"
    language_model = "codellama-7b-it"

    latest_date = None
    latest_file = None

    # Iterate over all files in the directory
    for filename in os.listdir(output_path):
        # Split the file name into parts
        parts = filename.split('_')

        # Check if the session name and language model match
        if ('test' not in filename and 'example' not in filename) \
            and (parts[0] == session_name or parts[0] + "_" + parts[1] == session_name) and language_model in parts[4]:
            # Parse the date and time from the file name
            date_time_str = parts[2] + ' ' + parts[3][:-3] + ":" + parts[3][-2:]
            date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M')

            if latest_date is None or date_time > latest_date:
                latest_date = date_time
                latest_file = filename

    return output_path + latest_file

def delete_directory(path="tmp/"):
    """
    This function deletes all files and directories in the given path but keeps the directory itself.
    :param path: str, path of the directory
    """
    # Check if the path exists
    if not os.path.exists(path):
        return

    # Check if the path is a directory or a file
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove dir and all contains
            except Exception as e:
                pass
    os.system(f'rm -r {path}* > /dev/null 2>&1')


if __name__ == "__main__":
    path_to_jsons = "workflow_automation/jsons/code_snippets_2024-06-03_17:28_codellama-7b-it.json"
    path_to_jsons = find_latest_jsons()
    
    automator = Automator(path_to_jsons)

    start = time.time()

    automator.run()

    end = time.time()
    print("Time taken: {} seconds, {} minutes, {} hours.".format(end - start, (end-start)/60, (end-start)/3600))

    delete_directory()
