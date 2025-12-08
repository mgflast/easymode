import os, shutil, json

version = "0.0.1"
license = "GNU GPL v3"

root = os.path.dirname(os.path.dirname(__file__))
settings_path = os.path.join(os.path.expanduser("~"), "easymode", "settings.txt")


def parse_settings():
    # If settings file not found, copy the one from core to the right location.
    if not os.path.exists(settings_path):
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        shutil.copy(os.path.join(root, "core", "settings.txt"), settings_path)

    try:
        with open(settings_path, 'r') as f:
            sdict = json.load(f)
    except Exception as e:
        shutil.copy(os.path.join(root, "core", "settings.txt"), settings_path)
        parse_settings()
        return

    # Read settings - if any parameters are missing, insert them.
    with open(os.path.join(root, "core", "settings.txt"), 'r') as f:
        default_settings = json.load(f)

    for key in default_settings:
        if key not in sdict:
            sdict[key] = default_settings[key]

    if sdict["MODEL_DIRECTORY"] == "" or not os.path.exists(sdict["MODEL_DIRECTORY"]):
        sdict["MODEL_DIRECTORY"] = os.path.join(os.path.expanduser("~"), "easymode")

    with open(settings_path, 'w') as f:
        json.dump(sdict, f, indent=2)

    return sdict


settings = parse_settings()


def edit_setting(key, value):
    global settings
    settings[key] = value
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    print(key, value)
