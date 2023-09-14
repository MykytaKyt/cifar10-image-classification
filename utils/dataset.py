import yaml


def get_path(path):
    try:
        a_yaml_file = open(path)
    except OSError:
        print('Wrong Path')
    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
    train = parsed_yaml_file['train']
    test = parsed_yaml_file['test']
    return train, test
