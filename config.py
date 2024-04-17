
from collections import namedtuple

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def print_config(config): 
    for k in config._asdict():
        print(k,config._asdict()[k])


def read_config(fname='config.txt'):
    config = {}
    with open(fname) as f:
        for line in f:
            try:
                key, value = line.strip().split('=')
                key = key.strip()
                value = value.strip()
                
                # Determine the appropriate data type and convert the value
                if value.lower() == 'true':
                    config[key] = True
                elif value.lower() == 'false':
                    config[key] = False
                elif str.isdigit(str(value)):
                    config[key] = int(value)
                elif is_float(str(value)):
                    config[key] = float(value)
                else:
                    ## throw error ValueError
                    raise ValueError
            except ValueError:
                print(f"Error in config.txt: Invalid value for key '{key}'")
            except Exception:
                print("Error in config.txt")
    
    MyDict = namedtuple('MyDict', config.keys())
    config = MyDict(**config)
    return config

