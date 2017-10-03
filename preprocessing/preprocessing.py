import re


def bag_of_words(s):
    lower_case_s = s.lower()
    alphanumeric_only_s = transform_to_alphanumeric_only(lower_case_s)
    return alphanumeric_only_s.split()


def transform_to_alphanumeric_only(s):
    sep = re.compile(r"[\W]")  # [\W] identifies all non-alphanumeric characters
    items = sep.split(s)  # split by those chars
    # join the strings, except it it's anyway a non-alphanumeric string
    return " ".join([item for item in items if item.strip() != ""])
