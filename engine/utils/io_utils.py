import pickle
import datetime


def get_string_datetime():
    now = datetime.datetime.now()
    if now.month < 10:
        month_string = "0" + str(now.month)
    else:
        month_string = str(now.month)
    if now.day < 10:
        day_string = "0" + str(now.day)
    else:
        day_string = str(now.day)
    yearmonthdate_string = str(now.year) + month_string + day_string
    return yearmonthdate_string


def write_list_to_file(my_list, path):
    with open(path, "w+") as f:
        for item in my_list:
            f.write("%s\n" % item)


def read_file_to_list(path):
    with open(path, "r") as f:
        x = f.readlines()
    return x


def write_pickle(data, path):
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return data


def convert_args_to_dict(args):
    return vars(args)


# Convert Tuple String to Integer Tuple
# Using tuple() + int() + replace() + split()
def convert_string_to_tuple(s):
    res = tuple(
        int(num)
        for num in s.replace("(", "").replace(")", "").replace(" ", "").split(",")
    )
    return res


def convert_to_tuple(value, n=2):
    """Converts a single value to a tuple of n identical values.

    Args:
      value: The value to be converted.
      n: The number of times to repeat the value.

    Returns:
      A tuple of n identical values.
    """

    return (value,) * n
