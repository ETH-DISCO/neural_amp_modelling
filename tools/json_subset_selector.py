import json

with open("data_config_300.json", "r") as fp:
    data_config = json.load(fp)

data_config["train"]["outputs"] = data_config["train"]["outputs"][:75]
json.dump(data_config, open("data_config_75.json", "w"), indent=4)