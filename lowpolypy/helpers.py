import os
import datetime


def get_output_name(input_name, suffix='lowpoly'):
    file, extension = os.path.splitext(input_name)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_file_name = "{}_{}_{}{}".format(file, suffix, now, extension)
    return output_file_name
