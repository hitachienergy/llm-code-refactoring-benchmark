This repository contains the code for the project "A Tool and Workflow for Benchmarking LLM-Based Code Improvements".

# Overview
`env/` contains the script to locally host and serve a language model with llama.cpp.

`tmp/` can be ignored, it will contain temporary artifacts which will also be cleaned up.

`sessions/` contains the Continue-style sessions used to query the language model.

`session_generator/` contains the code to create the above sessions.

`output/` contains the chat with the model, `query_LM.py` is the code that will take the above session and use it to query the language model.

`metrics/` contains the code to calculate the metrics using the output of the language model.

`workflow_automation/` contains the code that will replace the original code snippets with the updated ones and try to compile the code.

`visualization/` contains the code creates the plots following all other outputs. Note that outputs are saved at `workflow_automation/plots/`.

**Note**: When you clone the repository and want to run any script from it, make sure that you are in the root of the repository.

# Steps to run the code

## Environment setup

First, clone the repository.

Next, define environment variables for the project. Create a `.env` file in the root of the repository and add the following variables:

```bash
export CONDA_DIR=</path/to/conda>
export name=llama2
export HOST_IP="<IP>"
export PORT="<PORT>"
export API_URL="http://$HOST_IP:$PORT/v1/chat/completions"
```

Replace `<IP>` and `<PORT>` with the IP address and port of the server where your language model is hosted. Additionally, replace `</path/to/conda>` with the path to your conda installation.

Source the `.env` file to set the environment variables:

```bash
source .env
```

Next, create a conda environment using one of the environments provided in the `env/` directory. The environments contain the required libraries to run the code.


```bash
conda env create -f env/<environment_of_choice>.yml
```

Activate the environment by running:

```bash
conda activate llama2
```

## Running the code

Scripts need to be run in the following order:

### Session generator

Make sure that `sessions/code_snippets.md` is empty/clear. 
Add your snippets to `session_generator/snippets`
Add your prompts to `session_generator/prompt_formulation.md`, make sure each one starts with "- "

Run `session_generator/make_code_snippets.py`, this will populate `session/code_snippets.md`.

### Querying the LM
Whether you're hosting your LM or using a cloud-based LM, you need its IP address + PORT. `query_LM.py` will retrieve the corresponding API_URL from the environment variable set above.

Run `query_LM.py`, it will query the language model at the provided URL. When it's done, it will save the conversation to `output/`.

**Note**: For self hosting the LM, we used [llama.cpp](https://github.com/ggerganov/llama.cpp) which is easy to set up. The shell script used to host the language model is available in `env` . Make sure you change the path to the model as needed and the other parameters as you see fit. Also make sure that your environment fulfills all the llama.cpp requirements.

### Calculating metrics
After querying the LM, you can run `metrics/metrics.py` to calculate static metrics. 
The result will be saved to `workflow_automation/jsons`.

### Automation
`workflow_automation` takes care of replacing the original code snippets with the updated ones given by the LM and trying to compile and build the code. Simply run `automate.py`, the json will be updated at the end with the build and test metrics

### Visualization
Running `visualization/visualize.py` will generate plots in `workflow_automation/plots/`, namely heatmaps, tables and multibar plots.
Creating your own custom visualization is simple: simply add a function to the `Visualizer` class where you plot what you want to plot and add it to the list of running functions.

### Library Requirements:

	• lizard
	• radon
	• matplotlib
	• seaborn
	• pandas
	• numpy
	• tempfile
	• json
	• subprocess
	• time
	• shutil
	• difflib
	• re
	• requests
	• GitPython

**Note**: We recommend using a Linux machine to run the code, as there may be some formatting issues on Windows.