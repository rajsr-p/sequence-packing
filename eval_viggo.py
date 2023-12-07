import json
import copy
def get_fn_name(input):
    return input[:10]

def get_attributes(input):
    attributes = []
    idxs = [i for i in range(len(input)) if input.startswith('[', i)]
    for idx in idxs: 
        substr = input[:idx]
        space = max(substr.rfind(" "), substr.rfind("(")) + 1
        attributes.append(substr[space:])
    
    return attributes


# Put a output json file from 'endpoint_eval.py' here
with open('answers_gpt_4.json', 'r') as json_file:
    items = [json.loads(x) for x in json_file]

correct_fn = 0

for i in range(0, len(items)):
    finalstr = items[i]['input']
    input = finalstr[len("<TARGET_BEGIN>"):finalstr.find("<TARGET_END>")]
    output = finalstr[len("<REPR_BEGIN>")+finalstr.find("<REPR_BEGIN>"):finalstr.find("<REPR_END>")]
    model_output = items[i]["outputs"]

    actual_attr = get_attributes(output)
    model_attr = get_attributes(model_output)

    sorted_actual = copy.deepcopy(actual_attr)
    sorted_model = copy.deepcopy(model_attr)

    if (sorted_actual == sorted_model) and (get_fn_name(output) == get_fn_name(model_output)):
        correct_fn += 1
    
print(correct_fn)
print(len(items))
breakpoint()