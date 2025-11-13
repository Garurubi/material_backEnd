from request.material import SearchElementsRequest

def searchResultPaser(identifier: str, params: SearchElementsRequest, docs:list, filter:list[str]) -> str:
	"""검색 결과 String Parser"""
    
	if not docs:
		return "No materials found matching your criteria"
	
	results_md = (
        f"# Materials {identifier} Search Results\n\n"
        f"- **{identifier}**: {params.elements or 'All'}\n"
        f"- **Band gap range**: {params.filtering.band_gap_min} eV to {params.filtering.band_gap_max} eV\n"
        f"- **Stable Only**: {params.filtering.is_stable}\n\n"
        f"- **Showing up to {params.filtering.max_results} matches**\n\n"
    )

	# Truncate results to max_result
	docs = [doc.model_dump() for doc in docs][:params.filtering.max_results]

	for i, mat in enumerate(docs, 1):
		results_md += f"- **{i} Result {identifier}**\n"

		for key in filter:
			results_md += f"	- {key}: `{mat.get(key)}`\n"
		
		results_md += "\n"
	
	return results_md