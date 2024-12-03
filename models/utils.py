import re

def parse_loss_string(input_string, separator='_'):
    """
        Parse the input string to get the required losses and their weights
        The input string should be a underscore separated list of losses and weights
        For example: "1.0zls_1.0sdf_1.0normal_1.0div_1.0eikonal_1.0regexperts"
    """
    # Split the string by underscores
    parts = input_string.split(separator)

    numbers = []
    substrings = []

    # Iterate through the parts to separate numbers and substrings using regular expressions
    for part in parts:
        # Match numbers in the part
        num_match = re.match(r'([\d.]+)', part)
        if num_match:
            numbers.append(float(num_match.group(1)))  # Store the matched number as a float

        # Match substrings in the part
        substring_match = re.search(r'[a-zA-Z]+', part)
        if substring_match:
            substrings.append(substring_match.group(0))  # Store the matched substring

    return substrings, numbers


def build_loss_dictionary(required_loss_list, weights, model_type, full_loss_list):
    required_loss_dict = {}
    weight_dict = {}
    for loss_name, weight in zip(required_loss_list, weights):
        weight_dict[loss_name] = weight
        required_loss_dict[loss_name] = model_type
    for loss_name in full_loss_list:
        if loss_name not in required_loss_list:
            required_loss_dict[loss_name] = 'none'
            weight_dict[loss_name] = 0.0

    return required_loss_dict, weight_dict

