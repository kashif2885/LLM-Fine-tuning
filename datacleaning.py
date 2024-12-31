import json

# Load JSON data from file
with open('final_json_pak_const.json', 'r') as file:
    json_data = json.load(file)

# Extract qa_pairs from each section and store them in a list
result = []

for section in json_data.values():
    qa_pairs = section.get("qa_pairs", [])
    result.extend(qa_pairs)

# Save the result to a new JSON file or print it
with open('final2_json_pak_const.json', 'w') as output_file:
    json.dump(result, output_file, indent=4)

# Optionally, print the result
print(json.dumps(result, indent=4))