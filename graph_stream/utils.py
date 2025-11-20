import ast
import json
import re

from pydantic import BaseModel
from typing import Dict, List, Any
from enum import Enum


FILTER_OUT_DATA=["input", "messages"]


def parsing_node_output(node:str, data, ) -> Dict:
	""""""
	return_data = {}

	def serialize_value(value: Any) -> Any:
		"""재귀적으로 값을 직렬화"""
		# BaseModel (Pydantic)
		if isinstance(value, BaseModel):
			return value.model_dump(mode="json")
		elif isinstance(value, List):
			test = []
			for _value in value:
				if isinstance(_value, BaseModel):
					test.append( _value.model_dump(mode="json"))
			return test
		elif isinstance(value, Enum):
			return value.value
		# elif isinstance(value, Dict):
		# 	value.pop("messages", None)
		#	재귀 호출 코드 작성

		elif isinstance(value, (str, int, float, bool, type(None))):
			return value
		else:
			print(f"Skipping type: {type(value).__name__}")


	for key, value in data.items():
		if key not in FILTER_OUT_DATA:
			if isinstance(value, BaseModel):
				return_data[key] = value.model_dump(mode="json")
			elif isinstance(value, Dict):
				
				value.pop("messages", None)
				for k, v in value.items():
					if k == "messages":  # messages 필드 제외
						continue
					return_data[k] = serialize_value(v) 
				

	print("util 종료")
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