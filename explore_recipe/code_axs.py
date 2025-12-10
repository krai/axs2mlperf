import csv
from itertools import product
import hashlib
import time

# Function to preprocess the input query by removing a specific prefix
def preprocess_query(query, beginning_to_remove):
    # Check if the query starts with the specified prefix, and remove it
    if query.startswith(beginning_to_remove):
        return query[len(beginning_to_remove):].strip()
    else:
        # Raise an error if the prefix is missing
        raise ValueError(f"The command must begin with \"{beginning_to_remove}\".")

# Function to generate an entry name by replacing specific characters in the query
def get_entry_name(__query, prefix="explored_"):
    # Generate a 16-character hash of the input string
    hash_suffix = hashlib.sha256(__query.encode()).hexdigest()[:16]
    # Create the name
    name = prefix + hash_suffix
    return name

# Function to parse the query and store results of parsing into a csv file
def parse_and_store_commands(__query, beginning_to_remove, stored_newborn_entry=None, csv_file_name="parameters.csv", target_collection_name="experiments"):
    # Preprocess the query and remove 'dry_run' flags
    query = preprocess_query(__query, beginning_to_remove)
    query = query.replace(',dry_run+', '').replace(',dry_run-', '')

    # Split the query into individual parameters
    substrings = query.split(',')
    parameters = {} # Dictionary to store parsed parameters
    
    # Parse each substring into key-value pairs or flags
    for substring in substrings:
        # Key with multiple values: "x:=a:b:c"
        if ':=' in substring:
            key, values = substring.split(':=', 1)
            parameters[key] = values.split(':')
        # Key with a single value: "x=a"
        elif '=' in substring:
            key, value = substring.split('=', 1)
            parameters[key] = [value]
        # Flags with '+' or '-': "x+" or "x-"
        elif substring.endswith('+') or substring.endswith('-'):
            key = substring[:-1]
            value = substring[-1]
            parameters[key] = [value]
        # Tags with no value
        else:
            parameters[substring] = [""]

    # Add the target collection name to the parameters
    if target_collection_name:
        parameters["collection_name"] = [ target_collection_name ]

    # Generate combinations of parameter values
    headers = list(parameters.keys())
    values = list(parameters.values())
    combinations = list(product(*values))

    # Write the parameter combinations to a CSV file
    csv_path = stored_newborn_entry.get_path(csv_file_name)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(combinations)
    
    return csv_path

# Function to select which of the available commands to execute
def select_combination_to_execute(combinations, results, headers):
    if len(combinations) != len(results):
        raise ValueError("The number of combinations and results must match")

    selection = None
    # Loop through the combinations and select the first one that has not been executed
    # This simple implementation does not use any criteria for selection
    for index, result in enumerate(results):
        if result is None:
            selection = index
            break
    return selection

# Function to extract the result from the entry
def extract_result(new_entry):
    result = new_entry.get_name()
    return result

# Function to retrieve results of parsing, combine and execute commands
def retrieve_and_execute_commands(csv_path, explore_timeout_s, newborn_entry=None, __entry__=None, dry_run=False):
    # Read headers and combinations from the csv file
    with open(csv_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader)
        combinations = [row for row in reader]

    # Create a new list to store the results
    results = [None] * len(combinations)

    # Prepare and execute commands based on the combinations
    cmd_list = []
    query_list = []

    selection = __entry__.call("select_combination_to_execute", [ combinations, results, headers ], {})
    while selection is not None:
        combination = combinations[selection]
        # Map parameter names to their corresponding values
        config_cmd = dict(zip(headers, combination))

        # Construct the command with the parameters
        cmd_tag_list = []
        cmd_tag_collection = ''
        for key, value in config_cmd.items():
            # Handle flags ending with '+' or '-'
            if value and value[0] in ("+", "-"):
                cmd_tag_list.append(f"{key}{value[0]}")
            # Add key-value pairs to the command
            elif value or value == 0:
                cmd_tag_list.append(f"{key}={value}")
            # Add keys without values as tags
            else:
                cmd_tag_list.append(key)

        # Add an iteration number to the command
        cmd_tag_list.append(f"iteration={selection}")        

        # Construct the query
        new_query = ','.join(cmd_tag_list)
        query_list.append(new_query)

        # Construct the full command string
        cmd = f"axs byquery {new_query}{cmd_tag_collection}"
        cmd_list.append(cmd)

        # Escaping '\' to be sure that a proper command is passed into AXS
        cmd = cmd.replace("\"","\\\"")

        # Print the command in dry-run mode; execute it otherwise
        if dry_run:
            print(new_query)
            results[selection] = new_query
        else:
            new_entry = __entry__.get_kernel().byquery(new_query)
            result = __entry__.call("extract_result", [ new_entry ], {})
            new_entry.plant("performance_result", result)
            completed = new_entry.get("__completed")
            new_entry.save(completed=completed)
            results[selection] = result
        
        selection = __entry__.call("select_combination_to_execute", [ combinations, results, headers ], {})

        if explore_timeout_s > 0 and selection is not None:
            time.sleep(explore_timeout_s)

    # Store the list of commands in the newborn entry and save it
    newborn_entry.plant("query_list", query_list)
    newborn_entry.plant("cmd_list", cmd_list)
    newborn_entry.plant("results", results)
    newborn_entry.save()
