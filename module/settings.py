import yaml

class SettingsObject(dict):
    # __class__ = dict
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries to Struct objects
                setattr(self, key, SettingsObject(**value))
            else:
                setattr(self, key, value) # set attribute
        # self.__class__ = dict
    def __repr__(self):
        return str(self.__dict__)
    
    def __str__(self):
        return str(self.__dict__)
    
    def as_dict(self):
        return self.__dict__


def load_settings(config_file, settings_type = "default") -> dict:
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    if settings_type == 'default':
        settings = config['default_settings']
    elif settings_type == 'custom':
        settings = config['custom_settings']
    else:
        raise ValueError("No settings found. Choose 'default' or 'custom'.")
    return SettingsObject(**settings)