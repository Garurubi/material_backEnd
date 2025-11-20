import json
from langgraph.types import Command

from .utils import parsing_node_output

async def stream_graph_events(graph, config, inputs, is_resume):
	"""
	LangGraph astream_events 기반 전체 노드의 시작/종료 이벤트 스트리밍
	"""
	 # 중복 node 방지
	emitted_node_start = set()
	emitted_node_end = set()   

	# Interrupt 판단
	if is_resume:
		stream_input = Command(resume=inputs)
	else:
		stream_input = {"messages": [("user", inputs)]}

	async for event in graph.astream_events(stream_input, config,  subgraphs=True, stream_mode="updates"):

		event_type = event["event"]
		metadata = event.get("metadata", {})
		data = event.get("data", {})

		node = metadata.get("langgraph_node")

		if event_type == "on_chain_start" and node:
			if node not in emitted_node_start:
				emitted_node_start.add(node)
				yield (
					"event: node\n"
					f"data: {json.dumps({'node': node, "status": "start"})}\n\n"
				)
			continue

		if event_type == "on_chain_end" and node:
			if node not in emitted_node_end:
				emitted_node_end.add(node)
				parse_data = parsing_node_output(node, data)

				yield (
					"event: node\n"
					f"data: {json.dumps({'node': node, "status": "end", "data": parse_data})}\n\n"
				)
				continue

		if event_type == "on_chain_error":
			error = data.get("error", "Unknown error")
			yield (
				"event: error\n"
				f"data: {json.dumps({'error': str(error), "target": "node"})}\n\n"
			)


async def sse_event_generator(graph, config, inputs):
	""" SSE 이벤트 제너레이터 - astream_events 기반 """
	thread_id = config.get("configurable", {}).get("thread_id")
	yield f"event: stream\ndata: {json.dumps({'thread_id': thread_id, "status": "start"})}\n\n"
	
	is_resume = thread_id in graph.checkpointer.storage
	
	# if is_resume:
	# 	yield f"event: graph_resume\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
	
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
		
		try:
			graph.checkpointer.storage.pop(thread_id, None)
		except AttributeError:
			pass

		yield f"event: stream\ndata: {json.dumps({'thread_id': thread_id, "status": "end", 'final_report': final_report})}\n\n"
					
	except Exception as e:
		yield f"event: error\ndata: {json.dumps({'error': str(e), "target": "stream"})}\n\n"