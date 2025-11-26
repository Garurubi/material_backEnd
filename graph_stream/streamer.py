import orjson
from langgraph.types import Command

from .utils import parsing_node_output
import uuid


async def stream_graph_events(graph, config, inputs, is_resume):
	"""
	LangGraph astream_events 기반 전체 노드의 시작/종료 이벤트 스트리밍
	"""
	# 중복 node 방지
	node_map = {}

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

		if not node or node in ["supervisor_agent", "re_question"]: continue
		
		sse_type = "node" if "_agent" not in node else "agent"

		# 노드 진입
		if event_type == "on_chain_start" and node:
			if node not in node_map.keys():
				node_map[node] = uuid.uuid4()
				yield (
					f"event: {sse_type}\n"
					f"data: {orjson.dumps({'node_id': node_map[node],'node': node, "status": "start"}).decode('utf-8')}\n\n"
				)
				continue
		
		# 노드 종료
		if event_type == "on_chain_end" and node:
			try:
				if node in node_map.keys():
					node_id = node_map.pop(node)
					parse_data = parsing_node_output(node, data)

					yield (
						f"event: {sse_type}\n"
						f"data: {orjson.dumps({'node_id': node_id, 'node': node, "status": "end", "data": parse_data}).decode('utf-8')}\n\n"
					)
					continue
			except ValueError:
				pass
		
		# 에러 감지
		if event_type == "on_chain_error":
			error = data.get("error", "Unknown error")
			yield (
				"event: error\n"
				f"data: {orjson.dumps({'error': str(error), "target": "node"}).decode('utf-8')}\n\n"
			)
		
		
async def sse_event_generator(graph, config, inputs):
	""" SSE 이벤트 제너레이터 - astream_events 기반 """
	thread_id = config.get("configurable", {}).get("thread_id")
	
	yield f"event: stream\ndata: {orjson.dumps({'thread_id': thread_id, "status": "start"}).decode('utf-8')}\n\n"
	
	is_resume = thread_id in graph.checkpointer.storage
	
	try:
		async for event_msg in stream_graph_events(graph, config, inputs, is_resume):
			yield event_msg
			
		# 최종 상태 확인
		final_state = await graph.aget_state(config)
		
		# Interrupt 체크
		if final_state.next and final_state.tasks:
			interrupt_msg = final_state.tasks[0].interrupts[0].value

			yield f"event: interrupt\ndata: {orjson.dumps({'message': interrupt_msg}).decode('utf-8')}\n\n"
			return
		
		# 정상 종료: 더 이상 실행할 노드가 없고 interrupt도 없음
		final_report = final_state.values.get("final_report", final_state.values)
		
		try:
			graph.checkpointer.storage.pop(thread_id, None)
		except AttributeError:
			pass

		yield f"event: stream\ndata: {orjson.dumps({'thread_id': thread_id, "status": "end", 'final_report': final_report}).decode('utf-8')}\n\n"
					
	except Exception as e:
		yield f"event: error\ndata: {orjson.dumps({'error': str(e), "target": "stream"}).decode('utf-8')}\n\n"