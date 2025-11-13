from typing import Optional, List, Union, Literal
from pydantic import Field, AnyUrl, BaseModel

class FilterRequest(BaseModel):
	band_gap_min: float = Field(
		default=0.0,
		description="Lower bound for band gap filtering in eV. Materials with band gaps below this value will be excluded."
	)

	band_gap_max: float = Field(
		default=10.0,
		description="Upper bound for band gap filterfing in eV. Materials with band gaps above this value will be excluded."
	)

	is_stable: bool = Field(
		default=False,
		description="If True, only returns materials that are thermodynamically stable (energy above hull = 0)."
			" If False, returns all materials."
	)

	max_results: int = Field(
		default=10,
		ge=1,
		le=50,
		description="Maximum number of result to return. Must be between 1 and 50."
	)

class SearchElementsRequest(BaseModel):
	"""Material Project"""
	elements: Optional[List[str]] = Field(
		default=None,
		description="List of element symbols to fillter by (e.g., ['Si', 'O']). If None, searches across all elements."
	)

	filtering: FilterRequest = Field(default_factory=FilterRequest)