import sys

def get_total_size(obj, seen=None):
    """Recursively calculates the size of an object in bytes, including referenced objects."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Already counted this object
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        size += sum(get_total_size(k, seen) + get_total_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(get_total_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)

    return size

if __name__ == "__main__":
    # Example Usage:
    my_list = [1, "hello", [2, 3]]
    total_size = get_total_size(my_list)
    print(f"Total size of my_list: {total_size} bytes")

    my_dict = {"a": 1, "b": "world"}
    total_size_dict = get_total_size(my_dict)
    print(f"Total size of my_dict: {total_size_dict} bytes")

    class MyClass:
        def __init__(self):
            self.data = [1, 2, 3]

    my_instance = MyClass()
    total_size_instance = get_total_size(my_instance)
    print(f"Total size of my_instance: {total_size_instance} bytes")