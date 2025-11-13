from typing import Optional, List, Union, Literal
from pydantic import Field, AnyUrl, BaseModel


class CatalysisReactantsRequest(BaseModel):
    """Catalysis Reactant Request 파라미터"""

    reactant: str = Field(
        ..., 
        description="Reactant species to search for, provided as a string (e.g., 'H2')."
    )

    product: Optional[str] = Field(
        None, 
        description="Optional product species to search for, provided as a string (e.g., 'OH', 'OHstar')."
    )

    order: Literal["first", "last"] = Field(
        "first", 
        description="Pagination direction: 'first' retrieves results from the beginning, 'last' retrieves results from the end."
    )
    
    max_results: int = Field(
        default=10, 
        ge=1, 
        le=100, 
        description="Maximum number of results to return. Must be between 1 and 100."
    )

