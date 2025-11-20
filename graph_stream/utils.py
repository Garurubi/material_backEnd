import ast
import json
import re

from pydantic import BaseModel
from typing import Dict, List


FILTER_OUT_DATA=["input", "messages"]


def parsing_node_output(node:str, data:Dict, ) -> Dict:
	""""""
	return_data = {}

	for key, value in data.items():
		if key not in FILTER_OUT_DATA:
			if isinstance(value, BaseModel):
				return_data[key] = value.model_dump(mode="json")
			elif isinstance(value, Dict):
				value.pop("messages", None)
				return_data[key] = value

		
	return return_data

# if isinstance(data, BaseModel):
# 		parse_data = data.__dict__
# 		for key, value in parse_data.items():
# 			parse_data[key] = str(value)


# for key, value in parse_data.items():
# 			print(key, end=" -> ")
# 			print(value)
# 			print(type(value))
# 			print(json.dumps({key: str(value)}))