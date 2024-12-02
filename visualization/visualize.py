import sys
import json
import os
import time
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

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
                continue
    os.system(f'rm -r {path}* > /dev/null 2>&1')
    
def count_prompts():
    path = "session_generator/prompt_formulation.md"
    count = 0
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("- "):
                count += 1
    return count

def count_code_snippets():
    path = "session_generator/snippets/"
    return len(os.listdir(path))

def original_numb_of_prompt_snippet_combinations():
    num_prompts = count_prompts()
    num_snippets = count_code_snippets()
    return num_prompts * num_snippets

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

def get_original_metrics(master_df):
    # here i need to go through all original code snippets
    # calculate the metrics for each of them and compile them into one
    df = master_df.copy()

    # series of the original code
    og_code = df["original_code"]

    sys.path.append("metrics/")
    from metrics import analyze_csharp_code, calculate_cyclomatic_complexity_one
    
    # Get unique elements in the original_code series
    unique_codes = og_code.unique()

    # Initialize an empty list to store the results
    results = []

    # Iterate over each unique code
    for code in unique_codes:
        # Analyze the code and get a dictionary as output
        result = analyze_csharp_code(code)
        
        # Calculate cyclomatic complexity for the list of distinct elements
        complexities = calculate_cyclomatic_complexity_one(list(code)) # list of 1 element
        
        # Calculate the mean of the complexities
        mean_complexity = complexities[0] if complexities[0] else -1
        
        # Add the mean complexity to the result dictionary
        result["cyclomatic_complexity"] = mean_complexity

        results.append(result)

    # Initialize a defaultdict to store the sum of each key
    sum_dict = defaultdict(int)

    # Initialize a defaultdict to store the count of each key
    count_dict = defaultdict(int)

    # Iterate over each dictionary in results
    for result in results:
        # Iterate over each key-value pair in the dictionary
        for key, value in result.items():
            # Add the value to the sum of the key
            sum_dict[key] += value
            # Increment the count of the key
            count_dict[key] += 1

    # Initialize an empty dictionary to store the mean of each key
    mean_dict = {}

    # Iterate over each key in sum_dict
    for key in sum_dict:
        # Calculate the mean of the key and store it in mean_dict
        mean_dict["original_" + key] = sum_dict[key] / count_dict[key]

    final_df = pd.DataFrame([mean_dict])
    # final_df.columns = final_df.columns.str

    delete_directory()

    return final_df


class Visualizer:
    def __init__(self, path_to_jsons) -> None:
        self.path_to_db = path_to_jsons

    def __add_prompt_id(self, df):
        # Create a dictionary to map prompts to IDs
        prompt_to_id = {prompt: i for i, prompt in enumerate(df['prompt'].unique())}
        
        # Create a new column 'prompt_id' by mapping the 'prompt' column to its corresponding ID
        df['prompt_id'] = df['prompt'].map(prompt_to_id)

        # Taken from workflow_automation/prompt/prompt_keywords.json
        keywords_dict = {
            0: "Bugs, Fix",
            1: "Rewrite, Efficiency",
            2: "Refactor, Readability",
            3: "Refactor, CodeBlock",
            4: "Improve, SufficientQuality",
            5: "Bugs, CodeBlock",
            6: "Rewrite, CodeBlock",
            7: "Improve, BackwardCompatible"
        }

        # Replace the numbers with the corresponding keywords
        df['prompt_id'] = df['prompt_id'].map(keywords_dict)
        
        return df, prompt_to_id

    def __save_dict_to_json(self, dictionary, filename = "prompt_mapping.json", path="workflow_automation/plots/prompt/"):
        try:
            with open(path + filename, 'w', encoding='utf-8') as file:
                json.dump(dictionary, file)
        except Exception as e:
            print(f"An error occurred: {e}")

    def print_df(self, df):
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(df)

    def read_json(self, path_to_json):
        try:
            with open(path_to_json, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                return data
        except FileNotFoundError:
            print(f"File not found: {path_to_json}")
            sys.exit()

    def clean_prompts_column(self, df):
        df['prompt'] = df['prompt'].str.replace('\n', '')
        df['prompt'] = df['prompt'].str.replace('```', 'Given the above C# code your task is to find and fix bugs in it.')
        return df

    def drop_columns_with_keywords(self, df, keywords):
        cols_to_remove = [col for col in df.columns if any(keyword in col for keyword in keywords)]
        df = df.drop(columns=cols_to_remove)
        return df

    def drop_columns_without_keywords(self, df, keywords):
        cols_to_remove = [col for col in df.columns if not any(keyword in col for keyword in keywords)]
        df = df.drop(columns=cols_to_remove)
        return df

    def apply_mean_normalization(self, df):
        # Select only the numerical columns
        numerical_df = df.select_dtypes(include=['number'])
        numerical_df = numerical_df.drop(['pass_rate'], axis=1)
        
        # Apply mean normalization to the numerical columns
        normalized_df = (numerical_df - numerical_df.mean()) / numerical_df.std()
        
        # Drop non-finite values
        normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan).dropna()

        # Update the original dataframe with the normalized columns
        df.update(normalized_df.round(1).astype('int64'))

        return df
    
    def apply_min_max_normalization(self, df):
        # Select only the numerical columns
        numerical_df = df.select_dtypes(include=['number'])
        numerical_df = numerical_df.drop(['pass_rate'], axis=1)
        
        # Apply min-max normalization to the numerical columns
        normalized_df = (numerical_df - numerical_df.min()) / (numerical_df.max() - numerical_df.min())
        
        # Drop non-finite values 
        normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan).dropna()

        # Update the original dataframe with the normalized columns
        df.update(normalized_df.round(1).astype('int64'))
        
        return df

    def plot_prompt_metrics_table(self, df, groupby='prompt', type="abs", savepath="workflow_automation/plots/table/"):
        def create_table_abs(df, groupby='prompt'):
            assert((groupby == 'prompt' or groupby == "prompt_id"))

            # dont keep columns with min/max/mean/code in their name, otherwise they'll get plotted
            df = self.drop_columns_with_keywords(df, ["min", "max", "mean"])#, "code"])
            
            # Get the first row with pass_rate = 100%, then we'll take all the original features and use them. 
            # if there's no 100% pass rate, should havea function that extract all original code snippets, calculates original metrics.. etc 
            try:
                #Actually this is a dataframe
                original_row = df[df['pass_rate'] == 100]
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.iloc[0]
            except Exception as e:
                #Actually this returns a dataframe
                original_row = get_original_metrics(df)
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.iloc[0]
            

            # Get the original features
            original_features = original_row.filter(regex='^original_')

            # Rename the original features to match the updated features
            original_features.index = original_features.index.str.replace('original_', 'updated_')

            # Add NaN values for distance metrics in original row
            original_features['pass_rate'] = float('nan')
            original_features['updated_levenshtein_distance'] = float('nan')
            original_features['updated_syntactic_similarity'] = float('nan')

            # Create a new dataframe with the original features as the first row
            table = pd.DataFrame(original_features).T

            # Group the dataframe by prompt and calculate the mean of the updated features and pass rate
            # df = df.groupby(["repo_name", groupby]).mean(numeric_only=True)
            grouped_df = df.groupby(groupby).mean(numeric_only=True)

            grouped_df = grouped_df.round(1)

            # Create a list to store all rows
            rows = [table]

            # For each group, get the updated features and append them to the table
            for prompt, group in grouped_df.iterrows():
                updated_features = group.filter(regex='pass_rate|^updated_')
                rows.append(pd.DataFrame(updated_features).T)

            # Concatenate all rows
            table = pd.concat(rows)

            # Set the index of the table to be the prompts, with 'Original' as the first index
            table.index = ['Original'] + list(grouped_df.index)

            # Remove 'updated_' from the column names
            table.columns = table.columns.str.replace('updated_', '')

            # Put pass rate first [its always last column currently]
            cols = table.columns.tolist()
            cols.insert(0, cols.pop(cols.index('pass_rate')))  # Move 'Pass Rate' from its current position to the first
            table = table[cols]

            return table
        
        def create_table(df, groupby='prompt', mode='percentage'):
            assert((groupby == 'prompt' or groupby == "prompt_id"))

            # dont keep columns with min/max/mean/code in their name, otherwise they'll get plotted
            df = self.drop_columns_with_keywords(df, ["min", "max", "mean"])#, "code"])

            # Get the first row with pass_rate = 100%
            try:
                #Actually this is a dataframe
                original_row = df[df['pass_rate'] == 100]
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.iloc[0]
            except Exception as e:
                #Actually this returns a dataframe
                original_row = get_original_metrics(df)
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.iloc[0]

            # Get the original features
            original_features = original_row.filter(regex='^original_')

            # Rename the original features to match the updated features
            original_features.index = original_features.index.str.replace('original_', '')

            # Add a NaN value for the pass_rate in the original row
            original_features['pass_rate'] = float('nan')
            original_features['levenshtein_distance'] = float('nan')
            original_features['syntactic_similarity'] = float('nan')

            # Create a new dataframe with the original features as the first row
            table = pd.DataFrame(original_features).T

            # Group the dataframe by prompt
            # grouped_df = df.groupby(["repo_name", groupby])
            grouped_df = df.groupby(groupby)

            # Create a list to store all rows
            rows = [table]

            # For each group, get the updated features and calculate (updated [- original]) / original
            for prompt, group in grouped_df:
                updated_features = group.filter(regex='^updated_').mean(numeric_only=True)
                original_features = group.filter(regex='^original_').mean(numeric_only=True)
                original_features.index = original_features.index.str.replace('original_', 'updated_')


                if mode == 'percentage':
                    difference = (((updated_features - original_features) / original_features) * 100).round(1)
                elif mode == 'factor':
                    difference = ((updated_features / original_features) * 100).round(1)

                # Add the pass_rate to the difference
                difference['pass_rate'] = group['pass_rate'].mean()
                
                # Remove 'updated_' from the column names
                difference.index = difference.index.str.replace('updated_', '')
                rows.append(pd.DataFrame(difference).T)

            # Concatenate all rows
            table = pd.concat(rows)

            # Set the index of the table to be the prompts, with 'Original' as the first index
            table.index = ['Original'] + list(grouped_df.groups.keys())

            # Put pass rate first [its always last column currently]
            cols = table.columns.tolist()
            cols.insert(0, cols.pop(cols.index('pass_rate')))  # Move 'Pass Rate' from its current position to the first
            table = table[cols]

            return table
        
        table_df = df.copy()
        table_df["pass_rate"] = table_df["pass_rate"] * table_df[groupby].nunique()

        # Get unique repo names
        repo_names = table_df['repo_name'].unique()

        for repo in repo_names:
            df_repo = table_df[table_df['repo_name'] == repo]

            # Here can have switch case for abs/normalized/percentage/factor
            match type:
                case "abs":
                    df_repo = create_table_abs(df_repo, groupby=groupby)

                case "normalized":
                    # Apply normalization
                    df_repo = self.apply_min_max_normalization(df_repo)
                    # df_repo = self.apply_mean_normalization(df_repo)

                    df_repo = create_table_abs(df_repo, groupby=groupby)


                case "percentage":
                    df_repo = create_table(df_repo, mode="percentage", groupby=groupby)

                case "factor":
                    df_repo = create_table(df_repo, mode="factor", groupby=groupby)

                case _:
                    print("Unexpected table type. Will default to absolute numbers.")
                    df_repo = create_table_abs(df_repo, groupby=groupby)


            # Split the column names on underscore
            df_repo.columns = df_repo.columns.str.replace('_', ' ')
            df_repo.columns = df_repo.columns.str.replace('original', 'og')
            df_repo.columns = df_repo.columns.str.replace('updated', 'up')

            # Plot the grouped dataframe as a table
            fig, ax = plt.subplots(1, 1, figsize=(20, 8))

            # Hide axes
            ax.axis('off')

            # Trying 2 values in each cell
            # df_repo = df_repo.astype(str) + '   |   ' + (df_repo + 10).astype(str)


            # Create table and set its position
            table = ax.table(cellText=df_repo.values,
                            colLabels=df_repo.columns,
                            rowLabels=df_repo.index,
                            cellLoc = 'center', 
                            loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 3)

            # Save the figure to the specified path
            plt.savefig(os.path.join(savepath, f'table_{repo}_{type}.png'))
            plt.close(fig)

    def plot_heatmap(self, df, groupby = 'prompt', savepath="workflow_automation/plots/heatmap/"):
        assert((groupby == 'prompt' or groupby == "prompt_id"))
        
        heat_df = df.copy()
        heat_df["pass_rate"] = heat_df["pass_rate"] / 100

        # Apply normalization
        heat_df = self.apply_min_max_normalization(heat_df)
        # heat_df = self.apply_mean_normalization(heat_df)

        # dont keep columns with min/max/mean in their name
        heat_df = self.drop_columns_with_keywords(heat_df, ["min", "max", "mean", "code"])

        # change all original/updated to percentages
        # Get the original and updated column names
        original_cols = [col for col in heat_df.columns if col.startswith('original_')]
        updated_cols  = [col.replace('original_', 'updated_') for col in original_cols]
        
        # Calculate the percentual difference for each pair of original and updated columns
        for original_col, updated_col in zip(original_cols, updated_cols):
            # Make sure that we only work on numerical columns
            if np.issubdtype(heat_df[original_col].dtype, np.number) and np.issubdtype(heat_df[updated_col].dtype, np.number):
                new_col_name = original_col.replace('original_', '')
                heat_df[new_col_name] = (heat_df[updated_col] - heat_df[original_col]) / (heat_df[original_col])# + 1e-7)


        # drop all cols with /original/updated in their names
        heat_df = self.drop_columns_with_keywords(heat_df, ["original", "updated"])

        # groupby prompt at the end
        df_grouped_mean = heat_df.groupby(["repo_name", groupby]).mean(numeric_only=True)
        
        # Get unique repo names
        repo_names = df['repo_name'].unique()

        for repo in repo_names:
            df_repo = df_grouped_mean.loc[repo]

            # plot stuff
            plt.figure(figsize = (16, 12))
            color = plt.get_cmap('BuGn')
            color.set_bad('lightblue') # NaN values
            sns_plot = sns.heatmap(df_repo, annot=True, cmap=color, vmin=-1, vmax=1)
            fig = sns_plot.get_figure()
            
            # Save the figure with the repo name in the filename
            fig.savefig(os.path.join(savepath, f'heatmap_{repo}.png'), dpi=100)

    def plot_multibar_distribution(self, df, groupby='prompt', savepath="workflow_automation/plots/multibar/"):
        assert((groupby == 'prompt' or groupby == "prompt_id"))
        
        multi_df = df.copy()
        

        multi_df = self.apply_min_max_normalization(multi_df)
        multi_df["pass_rate"] = multi_df["pass_rate"] * multi_df[groupby].nunique()
        multi_df["pass_rate"] = multi_df["pass_rate"]/100

        # dont keep columns with min/max/mean in their name, otherwise they'll get plotted
        multi_df = self.drop_columns_with_keywords(multi_df, ["min", "max", "mean",])

        grouped_df = multi_df.groupby(["repo_name", groupby]).mean(numeric_only=True)

         # Get unique repo names
        repo_names = df['repo_name'].unique()

        for repo in repo_names:
            df_repo = grouped_df.loc[repo]

            # Get the list of prompts
            prompts = df_repo.index

            # Get the list of features
            features = df_repo.columns

            # Set the width of the bars
            bar_width = 0.05

            # Define a list of colors
            colors = plt.cm.magma(np.linspace(0, 1, len(features)))

            # Sort features so that original and updated features are next to each other
            sorted_features = sorted(features, key=lambda x: (x.split('_')[-1], x))

            # Ensure "pass_rate" is always the last one
            sorted_features.remove("pass_rate")
            sorted_features.append("pass_rate")

            # For each pair of features
            for i in range(0, len(sorted_features), 2):
                # Set the figure size
                fig, ax = plt.subplots(figsize=(24, 12))

                # Compute the position of the bars
                position = [j + i * bar_width for j in range(len(prompts))]

                # Create the bars for the current pair of features
                for j in range(2):
                    if i + j < len(sorted_features):
                        ax.bar(position, df_repo[sorted_features[i + j]], width=bar_width, label=sorted_features[i + j]) #, color=colors[i + j])
                        position = [p + bar_width for p in position]

                # Add the legend
                ax.legend(loc='best', bbox_to_anchor=(0.7, 0.7))

                ## Set the position of the x ticks
                ax.set_xticks([r + bar_width for r in range(len(prompts))])
                ax.set_xticklabels(prompts)

                # Rotate labels
                ax.tick_params(axis='x', labelrotation=90)

                # Remove 'original_' or 'updated_' from the feature name if it starts with it
                feature_name = sorted_features[i].replace('original_', '').replace('updated_', '')

                # Save the plot
                safe_feature_name = feature_name.replace('/', '_')
                plt.savefig(os.path.join(savepath, f'multibar_{repo}_{safe_feature_name}.png'))
                plt.close(fig)

    def plot_summary_table_per_element(self, df, groupby, type="abs", savepath="workflow_automation/plots/language/"):
        assert(groupby == "language" or groupby == "prompt_id")

        def create_table_abs(df, groupby='prompt'):

            # dont keep columns with min/max/mean/code in their name, otherwise they'll get plotted
            df = self.drop_columns_with_keywords(df, ["min", "max", "mean"])#, "code"])
            
            # Get the first row with pass_rate = 100%, then we'll take all the original features and use them. 
            # if there's no 100% pass rate, should havea function that extract all original code snippets, calculates original metrics.. etc 
            try:
                #Actually this is a dataframe
                original_row = df[df['pass_rate'] == 100]
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.round(1).iloc[0]
            except Exception as e:
                #Actually this returns a dataframe
                original_row = get_original_metrics(df)
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.round(1).iloc[0]
            

            # Get the original features
            original_features = original_row.filter(regex='^original_')

            # Rename the original features to match the updated features
            original_features.index = original_features.index.str.replace('original_', 'updated_')

            # Add NaN values for distance metrics in original row
            original_features['pass_rate'] = float('nan')
            original_features['updated_levenshtein_distance'] = float('nan')
            original_features['updated_syntactic_similarity'] = float('nan')

            # Create a new dataframe with the original features as the first row
            table = pd.DataFrame(original_features).T

            # Group the dataframe by prompt and calculate the mean of the updated features and pass rate
            # grouped_df = df.groupby(["repo_name", groupby]).mean(numeric_only=True)
            grouped_df = df.groupby(groupby).mean(numeric_only=True)

            grouped_df = grouped_df.round(1)

            # Create a list to store all rows
            rows = [table]

            # For each group, get the updated features and append them to the table
            for prompt, group in grouped_df.iterrows():
                updated_features = group.filter(regex='pass_rate|^updated_')
                rows.append(pd.DataFrame(updated_features).T)

            # Concatenate all rows
            table = pd.concat(rows)

            # Set the index of the table to be the prompts, with 'Original' as the first index
            table.index = ['Original'] + list(grouped_df.index)

            # Remove 'updated_' from the column names
            table.columns = table.columns.str.replace('updated_', '')

            # Put pass rate first [its always last column currently]
            cols = table.columns.tolist()
            cols.insert(0, cols.pop(cols.index('pass_rate')))  # Move 'Pass Rate' from its current position to the first
            table = table[cols]

            return table
        
        def create_table(df, groupby='prompt', mode='percentage'):

            # dont keep columns with min/max/mean/code in their name, otherwise they'll get plotted
            df = self.drop_columns_with_keywords(df, ["min", "max", "mean"])#, "code"])

            # Get the first row with pass_rate = 100%
            try:
                #Actually this is a dataframe
                original_row = df[df['pass_rate'] == 100]
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.round(1).iloc[0]
            except Exception as e:
                #Actually this returns a dataframe
                original_row = get_original_metrics(df)
                original_row = self.drop_columns_with_keywords(original_row, ["code"])

                # Now its a row
                original_row = original_row.round(1).iloc[0]

            # Get the original features
            original_features = original_row.filter(regex='^original_')

            # Rename the original features to match the updated features
            original_features.index = original_features.index.str.replace('original_', '')

            # Add a NaN value for the pass_rate in the original row
            original_features['pass_rate'] = float('nan')
            original_features['levenshtein_distance'] = float('nan')
            original_features['syntactic_similarity'] = float('nan')

            # Create a new dataframe with the original features as the first row
            table = pd.DataFrame(original_features).T

            # Group the dataframe by prompt
            # grouped_df = df.groupby(["repo_name", groupby])
            grouped_df = df.groupby(groupby)

            # Create a list to store all rows
            rows = [table]

            # For each group, get the updated features and calculate (updated [- original]) / original
            for prompt, group in grouped_df:
                updated_features = group.filter(regex='^updated_').mean(numeric_only=True)
                original_features = group.filter(regex='^original_').mean(numeric_only=True)
                original_features.index = original_features.index.str.replace('original_', 'updated_')


                if mode == 'percentage':
                    difference = (((updated_features - original_features) / original_features) * 100).round(1)
                elif mode == 'factor':
                    difference = ((updated_features / original_features) * 100).round(1)

                # Add the pass_rate to the difference
                difference['pass_rate'] = group['pass_rate'].round(1).mean()
                
                # Remove 'updated_' from the column names
                difference.index = difference.index.str.replace('updated_', '')
                rows.append(pd.DataFrame(difference).T)

            # Concatenate all rows
            table = pd.concat(rows)

            # Set the index of the table to be the prompts, with 'Original' as the first index
            table.index = ['Original'] + list(grouped_df.groups.keys())

            # Put pass rate first [its always last column currently]
            cols = table.columns.tolist()
            cols.insert(0, cols.pop(cols.index('pass_rate')))  # Move 'Pass Rate' from its current position to the first
            table = table[cols]

            return table
        

        df = df.copy()
        df["pass_rate"] = df["pass_rate"] * df[groupby].nunique()
        df = self.apply_min_max_normalization(df)
        
        # Get unique elements
        elements = df[groupby].unique()

        for element in elements:
            if not isinstance(element, np.int64) and "// c++" in element: continue
            df_language = df[df[groupby] == element]

            # Here can have switch case for abs/normalized/percentage/factor
            match type:
                case "abs":
                    table_df = create_table_abs(df_language, groupby=groupby)

                case "normalized":
                    # Apply normalization
                    table_df = self.apply_min_max_normalization(df_language)
                    table_df = create_table_abs(table_df, groupby=groupby)

                case "percentage":
                    table_df = create_table(df_language, mode="percentage", groupby=groupby)

                case "factor":
                    table_df = create_table(df_language, mode="factor", groupby=groupby)

                case _:
                    table_df = create_table_abs(df_language, groupby=groupby)

            # Plot the grouped dataframe as a table
            fig, ax = plt.subplots(1, 1, figsize=(20, 8))

            # Hide axes
            ax.axis('off')

            # Create table and set its position
            table = ax.table(cellText=table_df.values,
                            colLabels=table_df.columns,
                            rowLabels=table_df.index,
                            cellLoc = 'center', 
                            loc='center')

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 3)

            # Save the figure to the specified path
            if not 'prompt' in savepath:
                element = element.replace('//', '').replace('/', '').replace(' ', '')
                plt.savefig(os.path.join(savepath, f'table_{element}_{type}.png'))
            else:
                plt.savefig(os.path.join(savepath, f'table_prompt_{element}_{type}.png'))
            plt.close(fig)

    def plot_per_category(self, df, keywords_to_keep = ["pass_rate", "cyclomatic_complexity", "LOC", "comment"], groupby="prompt_id", savepath="workflow_automation/plots/"):
        assert((groupby == "prompt_id") or (groupby == "repo_name") or (groupby == "language"))
        match groupby:
            case "prompt_id": 
                savepath = savepath + "prompt/"
            case "repo_name": 
                savepath = savepath + "repo/"
            case "language": 
                savepath = savepath + "language/"

        # might want to replace this by number of dataframe rows
        # nevermind, otherwise it goes beyond 100%
        # this is 480 / whatever
        THEORETICAL_MAX_ELEMENTS = original_numb_of_prompt_snippet_combinations() / df[groupby].nunique()
        # this is 381 / whatever
        # THEORETICAL_MAX_ELEMENTS = df.shape[0] / df[groupby].nunique()

        category_df = df.copy()

        

        # dont keep columns with min/max/mean/code in their name
        category_df = self.drop_columns_with_keywords(category_df, ["min", "max", "mean", "code", "comment/LOC"])

        # Update pass rates
        # Note that here we're completely changing the pass rate to a specific case
        category_df["pass_rate"] = category_df.groupby(groupby)[groupby].transform('count') / THEORETICAL_MAX_ELEMENTS

        # Group df and take means
        grouped_df = (category_df.groupby(groupby).mean(numeric_only=True)).round(1)

        # Keep interesting columns only
        grouped_df = self.drop_columns_without_keywords(grouped_df, keywords_to_keep)

        # Apply normalization
        # grouped_df = self.apply_min_max_normalization(grouped_df)
        # grouped_df = self.apply_mean_normalization(grouped_df)

        # Split the column names on underscore
        # grouped_df.columns = grouped_df.columns.str.replace('_', ' ')
        grouped_df.columns = grouped_df.columns.str.replace('original', 'og')
        grouped_df.columns = grouped_df.columns.str.replace('updated', 'up')

        # Add the number of elements to the DataFrame
        grouped_df['num_elements'] = category_df[groupby].value_counts()

        # Plot the grouped DataFrame as a table
        fig, ax = plt.subplots(1, 1, figsize=(40, 6))
        ax.axis('tight')
        ax.axis('off')

        # Create table and set its position
        table = ax.table(cellText=grouped_df.values,
                        colLabels=grouped_df.columns,
                        rowLabels=grouped_df.index,
                        cellLoc = 'center', 
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(20)
        table.scale(1, 3)

        grouped_df.to_csv(os.path.join(savepath, f'TABLE_{groupby}.csv'))
        plt.savefig(os.path.join(savepath, f'TABLE_{groupby}.png'))
        plt.close(fig)

        # Normalize for bar plot
        # grouped_df = self.apply_min_max_normalization(grouped_df)
        # grouped_df["pass_rate"] = grouped_df["pass_rate"] * 100

        # Dropping pass rate because it's not great to look at in a bar plot
        grouped_df = grouped_df.drop(['pass_rate', 'num_elements'], axis=1)


        # Plot a bar plot of the grouped DataFrame
        ax = grouped_df.plot(kind='bar', figsize=(12, 12), fontsize=16)

        # Add labels to the x and y axes
        ax.set_xlabel(groupby, fontsize=16)
        ax.set_ylabel("Metric values", fontsize=16)

    
        # Move the legend to the upper right and make it a bit smaller
        ax.legend(loc='upper right', prop={'size': 12})
        ax.tick_params(axis='x', labelrotation=30)

        # Add 'number of elements' to the plot
        for i, v in enumerate(category_df[groupby].value_counts()):
            ax.text(i, v + 0.2, str(v), ha='center', va='bottom', fontsize=12)

        # Add total elements to the plot
        total_elements = category_df[groupby].count()
        ax.text(len(df[groupby].value_counts()), total_elements + 0.2, f'Total: {total_elements}', ha='center', va='top', fontsize=28)

        #Save
        plt.savefig(os.path.join(savepath, f'BAR_{groupby}.png'))
        plt.close()


        # Plot heatmap
        # Apply normalization
        # category_df = self.apply_min_max_normalization(category_df)
        
        # change all original/updated to percentages
        # Get the original and updated column names
        # Define a dictionary for the column name mappings
        column_mappings = {
            'original_cyclomatic_complexity': 'original_cyclo',
            'updated_cyclomatic_complexity': 'updated_cyclo',
            'original_levenshtein_distance': 'original_edit_dist',
            'updated_levenshtein_distance': 'updated_edit_dist',
            'original_syntactic_similarity': 'original_syn_simi',
            'updated_syntactic_similarity': 'updated_syn_simi',
        }

        # Rename the columns
        category_df = category_df.rename(columns=column_mappings)

        original_cols = [col for col in category_df.columns if col.startswith('original_')]
        updated_cols  = [col.replace('original_', 'updated_') for col in original_cols]
        
        # Calculate the percentual difference for each pair of original and updated columns
        for original_col, updated_col in zip(original_cols, updated_cols):
            # Make sure that we only work on numerical columns
            if np.issubdtype(category_df[original_col].dtype, np.number) and np.issubdtype(category_df[updated_col].dtype, np.number):
                new_col_name = original_col.replace('original_', '')
                category_df[new_col_name] = ((category_df[updated_col] - category_df[original_col]) / (category_df[original_col])) * 100
        category_df["pass_rate"] = category_df["pass_rate"] * 100

        # drop all cols with /original/updated in their names
        category_df = self.drop_columns_with_keywords(category_df, ["original", "updated"])

        # groupby prompt at the end
        df_grouped_mean = (category_df.groupby(groupby).mean(numeric_only=True)).round(1)
        
        # plot stuff
        plt.figure(figsize = (20, 12))
        color = plt.get_cmap('BuGn')
        color.set_bad('lightblue') # NaN values
        sns.set(font_scale=1.5)
        sns_plot = sns.heatmap(df_grouped_mean, annot=True, cmap=color, vmin=-100, vmax=100)

        # Add labels to the x and y axes
        sns_plot.set_xlabel("Metrics", fontsize=16)

        fig = sns_plot.get_figure()
        
        # Save the figure with the repo name in the filename
        fig.savefig(os.path.join(savepath, f'HEATMAP_{groupby}.png'), dpi=100)

    def plot_describe(self, df, savepath="workflow_automation/plots/"):
        df = df.copy().round(1)
        df.columns = df.columns.str.replace('original', 'og')
        df.columns = df.columns.str.replace('updated', 'up')
        df.columns = df.columns.str.replace('comment', 'cmnt')
        # Define a dictionary for the column name mappings
        column_mappings = {
            'og_cyclomatic_complexity': 'og_cyclo',
            'up_cyclomatic_complexity': 'up_cyclo',
            'og_levenshtein_distance': 'edit_dist',
            # 'up_levenshtein_distance': 'up_levenshtein',
            'og_syntactic_similarity': 'syn_simi',
            # 'up_syntactic_similarity': 'up_syntactic',
        }

        # Rename the columns
        df = df.rename(columns=column_mappings)
        df = df.drop(["up_levenshtein_distance", "up_syntactic_similarity"], axis = 1)

        # Get the describe DataFrame
        describe_df = df.describe().round(1)

        # Plot the describe DataFrame as a table
        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=describe_df.values,
                        colLabels=describe_df.columns,
                        rowLabels=describe_df.index,
                        cellLoc = 'center', 
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 3)

        describe_df.to_csv(os.path.join(savepath, "summary_table.csv"))
        plt.savefig(os.path.join(savepath, "summary_table.png"))
        plt.close(fig)

    def rearrange_dataframe_columns(self, df):
        '''
        This function rearranges the dataframe columns such that all original_ and updated_ pairs are next to each other
        The rest of the columns will come after
        '''
        original_cols = [col for col in df.columns if col.startswith('original_')]
        updated_cols = [col.replace('original_', 'updated_') for col in original_cols]
        other_cols = [col for col in df.columns if col not in original_cols and col not in updated_cols]

        # Check if all corresponding updated_ columns exist
        # if not all(col in df.columns for col in updated_cols):
        #     raise ValueError("Some original_ columns do not have corresponding updated_ columns")

        # Reorder the dataframe
        new_order = other_cols + [col for pair in zip(original_cols, updated_cols) for col in pair]
        df = df[new_order]

        return df
    
    def add_aggregate_metrics(self, df, groupby='prompt'):
        assert((groupby == 'prompt' or groupby == "prompt_id"))
        
        # Keep only the numerical columns
        df_nums = df.select_dtypes(['number']).copy()
        df_nums["repo_name"] = df["repo_name"]

        agg = df_nums.groupby("repo_name").agg(['min','max', 'mean'])
        
        # Adding the aggregate features to my df
        for col in agg.columns:
            df[f'{col[0]}_{col[1]}'] = df['repo_name'].map(agg[col])
            # df[f'{col}_min'] = agg.loc['min', col]
            # df[f'{col}_max'] = agg.loc['max', col]
            # df[f'{col}_mean'] = agg.loc['mean', col]

        # Add pass rate
        prompt_counts = df.groupby(["repo_name", groupby]).size()
        prompt_counts = ((prompt_counts / count_code_snippets()) * 100).round(1)

        # old way
        # df['pass_rate'] = df[groupby].map(prompt_counts)

        # new way 1
        # Map the pass_rate
        # df['pass_rate'] = df.apply(lambda row: prompt_counts.loc[row["repo_name"], row[groupby]], axis=1

        # new way 2
        # Set the DataFrame index to match the multi-index of prompt_counts
        df.set_index(["repo_name", groupby], inplace=True)

        # Map the pass_rate
        df['pass_rate'] = df.index.map(prompt_counts)

        # Reset the index
        df.reset_index(inplace=True)

        return df

    def run(self):
        data = self.read_json(self.path_to_db)
        df = pd.DataFrame(data)
        df = self.clean_prompts_column(df)
        df, prompt_to_id = self.__add_prompt_id(df)
        self.__save_dict_to_json(prompt_to_id)

        # self.print_df(df.describe())
        df = self.rearrange_dataframe_columns(df)
        self.plot_describe(df)
        
        df = self.add_aggregate_metrics(df, groupby='prompt_id')
        df = self.rearrange_dataframe_columns(df)

        TABLE_TYPES = ["abs", "normalized", "percentage", "factor"]
        # for type in TABLE_TYPES:
        #     self.plot_prompt_metrics_table(df, type=type, groupby='prompt_id')
        #     delete_directory()
        #     self.plot_summary_table_per_element(df, type=type, groupby='language', savepath="workflow_automation/plots/language")
        #     delete_directory()
        #     self.plot_summary_table_per_element(df, type=type, groupby='prompt_id', savepath="workflow_automation/plots/prompt")

        # self.plot_heatmap(df, groupby='prompt_id')
        # self.plot_multibar_distribution(df, groupby='prompt_id')
        # delete_directory()

        STAR_CATEGORIES = ["prompt_id", "language", "repo_name"]
        for category in STAR_CATEGORIES:
            self.plot_per_category(df, groupby=category)
        

        # self.print_df(df.columns)


if __name__ == "__main__":
    delete_directory()
    
    # last json with 3 snippets.
    path_to_jsons = "workflow_automation/jsons/code_snippets_2024-06-03_17:28_codellama-7b-it.json"
    
    path_to_jsons = find_latest_jsons()

    visualizer = Visualizer(path_to_jsons)

    start = time.time()

    visualizer.run()

    end = time.time()
    print("Time taken: {} seconds, {} minutes, {} hours.".format(end - start, (end-start)/60, (end-start)/3600))

    delete_directory()
