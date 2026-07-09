from typing import Any

def matches_filter(metadata: dict[str, Any], filter_dict: dict[str, Any])-> bool:
    """
    Returns True if metadata satisfies every key-value pair in filter_dict (AND logic).
    A record missing a filtered key is treated as not matching.
    """

    for key, expected_value in filter_dict.items():
        if key not in metadata:
            return False
        if metadata[key] != expected_value:
            return False
    return True