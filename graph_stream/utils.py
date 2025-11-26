from pydantic import BaseModel
from typing import Dict, List, Any
from enum import Enum

FILTER_OUT_DATA=["input", "messages"]


def serialize_value(value: Any) -> Any:
	"""재귀적으로 값을 직렬화"""
	# BaseModel (Pydantic)
	if isinstance(value, BaseModel):
		return value.model_dump(mode="json")
	elif isinstance(value, List):
		serialize_list_data = []
		for _value in value:
			serialize_list_data.append(serialize_value(_value))
		return serialize_list_data
	
	elif isinstance(value, Enum):
		return value.value
	
	elif isinstance(value, Dict):
		value.pop("messages", None)
		return_data = []
		for k, v in value.items():
			return_data.append({k: serialize_value(v)})
		
		return return_data
	# isinstance(value, (str, int, float, bool, type(None))))
	else: 
		return value
	

def parsing_node_output(node:str, data, ) -> Dict:
	""""""
	return_data = {}

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
				
	return return_data