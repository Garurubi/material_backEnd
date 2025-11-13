import os
import logging
import requests
import json
from mcp.server.fastmcp import FastMCP

# 프로젝트 루트 추가(해당 파일 server 폴더로 이동 시 루트폴더 인식 필요)
# import sys
# from pathlib import Path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

# Custom
from request.catalysis import CatalysisReactantsRequest
from utils import catalysis_utils

# Setting logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("catalysis_hub_mcp")

# Catalysis-hub URL
CATALYSIS_URL = "https://api.catalysis-hub.org/graphql"

# Create the MCP server instance
mcp = FastMCP(
    name="catalysis_hub",
)

@mcp.tool()
async def search_reactions(params: CatalysisReactantsRequest) -> dict:
    """
    Search the Catalysis-Hub database for reactions matching the given reactant 
    and optional product parameters.

    Args:
        params (CatalysisReactantsRequest): 
            Query parameters including reactant, optional product, 
            pagination order, and maximum results.

    Returns:
        str: A JSON-formatted string containing a list of matching reaction records.
    """

    logger.info("Starting search_reactions query...")

    query = catalysis_utils.build_reactions_query(params)

    try:
        response = requests.post(CATALYSIS_URL, json={"query": query})
        response.raise_for_status()  
        
        response_data = response.json()
        data = response_data.get("data",{}).get("reactions",{}).get("edges",None)

        markdown_result = catalysis_utils.searchResultPaser(data)
        # markdown_result = data
        markdown_len = markdown_result.count(" Reaction:")

    except Exception as e:
        logger.error(e)
        return f"Error: 알 수 없는 오류가 발생했습니다. ({e})"
    result = {}
    result["result"] = markdown_result
    result["count"]=markdown_len
    # return markdown_result
    return result


if __name__ == "__main__":
    # print(f"Starting {mcp.name} server...")
    mcp.run(transport="stdio")