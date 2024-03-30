
from collections import namedtuple

def read_config():
    config = {}
    with open('config.txt') as f:
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
                elif '.' in value:
                    config[key] = float(value)
                else:
                    config[key] = int(value)
            except ValueError:
                print(f"Error in config.txt: Invalid value for key '{key}'")
            except Exception:
                print("Error in config.txt")
    
    MyDict = namedtuple('MyDict', config.keys())
    config = MyDict(**config)
    return config

