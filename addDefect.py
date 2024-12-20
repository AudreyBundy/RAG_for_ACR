import os
import subprocess
import json


def extract_and_analyze(data_entry, temp_dir="temp_code"):
    # Step 1: Create the .py file from old_hunk
    old_code = data_entry['old_hunk']
    filename = os.path.join(temp_dir, f"temp_{data_entry['ids'][0]}.py")

    os.makedirs(temp_dir, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(old_code)

    # Step 2: Run pytype on the generated file
    result = subprocess.run(['pytype', filename], capture_output=True, text=True)

    # Step 3: Check the output for errors
    errors = []
    if result.returncode != 0:
        errors = result.stderr.splitlines()


    # Step 4: Match errors with the change location
    matched_errors = []
    for error in errors:
        if any(data_entry['old'] in error for data_entry in data_entry['ids']):
            matched_errors.append({
                "error": error,
                # "location": data_entry['old']
            })

    return matched_errors


def analyze_data_file(data_file_path, output_file_path):
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for entry in data:
        # Step: Only process if language is Python
        if entry.get("lang") == "py":
            errors = extract_and_analyze(entry)
            if errors:
                entry['matched_errors'] = errors
        results.append(entry)

    # Write the results back to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    data_file_path = r"D:\GRADUATION\Nju\Dataset\MyDataSet\codereviewer_v1\filtered_data.json"
    output_file_path = "results.json"
    analyze_data_file(data_file_path, output_file_path)
