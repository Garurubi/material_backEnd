import os
import logging
import json
from textwrap import dedent
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mp_api.client import MPRester

# 프로젝트 루트 추가(해당 파일 server 폴더로 이동 시 루트폴더 인식 필요)
# import sys
# from pathlib import Path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# Custom
from request import material as request_material
from utils.matetial_utils import searchResultPaser

# Setting logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("materials_project_mcp")

# 환경 변수 LOAD
# load_dotenv()
MP_API_KEY = os.environ.get("MP_API_KEY")

SEARCH_FILED = [
        "material_id", "formula_pretty", "formation_energy_per_atom",
        "energy_above_hull", "decomposes_to", "symmetry",
        "band_gap", "volume", "density", "is_stable", "nsites"
    ]

# Create the MCP server instance
mcp = FastMCP(name="materials_project")

def _get_mp_rester() -> MPRester:
    """
    Initialize and return a MPRester session with the user's API key.

    Returns:
        MPRester: An authenticated MPRester instance for querying the Materials Project AIP.
    
    Note:
        If no API key is found in enviroment variables, attempts to initalize without key.
    """

    if not MP_API_KEY:
        logger.warning(
            "No MP_AIP_KEY found in enviroment. Attempting MPRester() without key."
        )
        return MPRester()
    
    return MPRester(MP_API_KEY)

@mcp.tool()
# async def search_element_materials(params: request_material.SearchElementsRequest) -> str:
async def search_element_materials(params: request_material.SearchElementsRequest) -> dict:
    """
    Search for materials in the Materials Project database using elements.

    This function allows searching for materials based on their elemental composition,
    band gap range, and thermodynamic stability. Results are return in a formatted
    JSON string.

    Args:
        params (MaterialSearchElementsRequest): 
            
            Search parameters including:

                    - elements (list[str], optional): List of element symbols to filter by.
                    - filtering (FilterRequest): Band gap range, stability, and result limits.
    
    Returns:
        str: A JSON-formatted string containing the search results.
    """
    
    with _get_mp_rester() as mpr:
        docs = mpr.materials.summary.search(
            elements= params.elements,
            band_gap=(params.filtering.band_gap_min, params.filtering.band_gap_max),
            is_stable=params.filtering.is_stable,
            fields=SEARCH_FILED
        )

    resutls_parser = searchResultPaser("Elements", params, docs, SEARCH_FILED)
    # resutls_parser = docs
    result_len  = resutls_parser.count(" Result ")
    result = {}
    result["result"] = resutls_parser
    result["count"] = result_len
    # return resutls_parser
    return result


if __name__ == "__main__":
    # print(f"Starting {mcp.name} server...")
    mcp.run(transport="stdio")
