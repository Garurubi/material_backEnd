from textwrap import dedent
from request.catalysis import CatalysisReactantsRequest
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("catalysis_utils")

def build_reactions_query(params: CatalysisReactantsRequest) -> str:
    """GraphiQL reataions 쿼리 생성 함수
    
    Args:
        params (CatalysisReactantsParam): Object containing search parameters.
    
    Returns:
        str: catalysis GraphiQL reactions query
    """
    
    pagination = f"{params.order}: {params.max_results}"
    
    query_args = [pagination, f'reactants: "{params.reactant}"']
    if params.product:
        query_args.append(f'products: "{params.product}"')
        
    query = f"""
    {{
        reactions({", ".join(query_args)}) {{
            edges {{
                node {{
                    id
                    chemicalComposition
                    surfaceComposition
                    facet
                    reactionEnergy
                    activationEnergy
                    reactionSystems{{
                    id
                    aseId
                    name
                    energyCorrection
                    systems{{
                        uniqueId
                        energy
                        Formula
                        natoms
                    }}
                    }}
                }}
            }}
        }}
    }}
    """
    return dedent(query)


def searchResultPaser(reactions: List[Dict]) -> str:
    """여러 reaction 결과를 Markdown 계층 구조로 변환 (토큰 절약용)"""
    
    if not reactions:
        return "No materials found matching your criteria"
    
    md_result = ""
    
    for idx, reaction_node in enumerate(reactions, 1):
        node = reaction_node['node']
        
        # Reaction 기본 정보
        md_result += f"**{idx}. Reaction:** {node['chemicalComposition']} (surface: {node['surfaceComposition']}, facet: {node['facet']})\n"
        md_result += f"- Reaction Energy: `{node['reactionEnergy']:.4f}` eV\n"
        activation = node.get('activationEnergy')
        md_result += f"- Activation Energy: `{activation:.4f}` eV\n" if activation is not None else "- Activation Energy: `None`\n"
        
        # Reaction 시스템 정보
        md_result += "  **Systems involved:**\n"
        for sys_idx, sys in enumerate(node.get('reactionSystems', []), 1):
            s = sys['systems']
            md_result += f"  {sys_idx}. {sys['name']}\n"
            md_result += f"     - Formula: `{s['Formula']}`\n"
            md_result += f"     - Energy: `{s['energy']:.4f}` eV\n"
            md_result += f"     - Natoms: `{s['natoms']}`\n"
        md_result += "\n"
    
    return md_result