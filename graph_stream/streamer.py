import asyncio
import json
from typing import Any
from langgraph.types import Command
from langchain_core.messages import BaseMessage


import asyncio
import json
from langgraph.types import Command


async def stream_graph_events(graph, config, inputs, is_resume):
	"""LangGraph astream_events를 사용(모든 노드 실행 추적)"""
	subgraph_stack = []
	seen_nodes = set()
	
	if is_resume:
		stream_input = Command(resume=inputs)
	else:
		stream_input = {"messages": [("user", inputs)]}
	
	async for event in graph.astream_events(stream_input, config,  subgraphs=True, stream_mode="updates"):
		
		event_type = event.get("event")
		name = event.get("name", "")
		metadata = event.get("metadata", {})
		
		# supervisor/placeholder 제외
		if "supervisor" in name.lower() or "placeholder" in name.lower():
			continue
		
		# 노드 시작
		if event_type == "on_chain_start":
			langgraph_node = metadata.get("langgraph_node")
			if langgraph_node:
				node_id = f"{langgraph_node}_{event.get('run_id', '')}"
				
				if node_id not in seen_nodes:
					seen_nodes.add(node_id)
					
					# 서브그래프 감지
					if "subgraph_" in langgraph_node or metadata.get("langgraph_path"):
						path = metadata.get("langgraph_path", [])
						if len(path) > 1: 
							if not subgraph_stack or subgraph_stack[-1] != langgraph_node:
								subgraph_stack.append(langgraph_node)
								yield f"event: subgraph_start\ndata: {json.dumps({'node': langgraph_node})}\n\n"
						else:
							yield f"event: node_start\ndata: {json.dumps({'node': langgraph_node})}\n\n"
					else:
						yield f"event: node_start\ndata: {json.dumps({'node': langgraph_node})}\n\n"
		
		# # 노드 종료
		# elif event_type == "on_chain_end":
		# 	langgraph_node = metadata.get("langgraph_node")
		# 	if langgraph_node:
		# 		if langgraph_node in subgraph_stack:
		# 			subgraph_stack.remove(langgraph_node)
		# 			yield f"event: subgraph_end\ndata: {json.dumps({'node': langgraph_node})}\n\n"
		# 		else:
		# 			yield f"event: node_end\ndata: {json.dumps({'node': langgraph_node})}\n\n"
		
		elif event_type == "on_chain_error":
			error = event.get("data", {}).get("error", "Unknown error")
			yield f"event: error\ndata: {json.dumps({'message': str(error)})}\n\n"


async def sse_event_generator(graph, config, inputs):
	""" SSE 이벤트 제너레이터 - astream_events 기반 """
	thread_id = config.get("configurable", {}).get("thread_id")
	yield f"event: graph_start\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
	
	is_resume = thread_id in graph.checkpointer.storage
	
	if is_resume:
		yield f"event: graph_resume\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
	
	try:
		async for event_msg in stream_graph_events(graph, config, inputs, is_resume):
			yield event_msg
			
		# 최종 상태 확인
		final_state = await graph.aget_state(config)
		
		# Interrupt 체크
		if final_state.next and final_state.tasks:
			interrupt_msg = final_state.tasks[0].interrupts[0].value
			yield f"event: interrupt\ndata: {json.dumps({'message': interrupt_msg})}\n\n"
			return
		
		# 정상 종료: 더 이상 실행할 노드가 없고 interrupt도 없음
		final_report = final_state.values.get("final_report", final_state.values)
		if isinstance(final_report, dict):
			final_report.pop("messages", None)
		
		try:
			graph.checkpointer.storage.pop(thread_id, None)
		except AttributeError:
			pass

		yield f"event: stream_end\ndata: {json.dumps({'final_report': final_report})}\n\n"
					
	except Exception as e:
		yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"