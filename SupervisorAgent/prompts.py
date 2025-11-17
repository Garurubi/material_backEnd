clarify_with_user_instructions="""You are an intelligent research-domain classification agent.
Your task is to analyze a user's scientific query message and produce a structured JSON 
that follows the ClarifyWithUser schema.

<Messages>
{messages}
</Messages>

Categories:
- Single-Atom Catalysts
- Photovoltaic Tandem Devices
- Biopolymer Materials
- Other

1. **Identify the intent of the user message**  
   - Analyze the user’s message and summarize the research intent.   

2. **Classify domain**
   - Choose exactly one domain from the four categories above.
   - Decide based on the main research topic of the message.  
   - If the domain is unclear or not in scope, use `"Other"`.

3. **Generate a clarification question**
   - Compose one short Korean question to confirm if the investigation should proceed in that domain.
   - If the domain is **NOT "Other"**,  
     → Generate a short, polite Korean question confirming if the investigation should proceed in that domain.  
       Example:  
       “단일 원자 촉매 분야로 분석을 진행할까요?”
   - If the domain is **"Other"**,  
     → Ask the user to specify one of the three available domains explicitly:  
       **Single-Atom Catalysts / Photovoltaic Tandem Devices / Biopolymer Materials**.  
       Example:  
       “연구 주제가 명확하지 않습니다. 단일 원자 촉매, 태양광 탠덤 소자, 바이오폴리머 소재와 관련해서 다시 질문해주세요.”
"""
paper_system_instructions = """Your MOST IMPORTANT RULE: **Do not change the mapping between input and output.**  

- Every input paper has a unique integer `paper_id` (1, 2, 3, …).  
- In the output, you MUST return the SAME `paper_id` for the SAME paper.
"""

paper_clarify_instructions = """You are a domain classifier for scientific papers.  
Given the title and abstract of multiple papers, classify each one into one of the following categories:

<Papers>
{papers}
</Papers>

Categories:
- Single-Atom Catalysts
- Photovoltaic Tandem Devices
- Biopolymer Materials
- Other

RULES:
- Base your decision ONLY on each paper’s title and abstract.  
- For each paper, choose exactly one category.
- If the paper is unrelated to all categories, answer "Other".  
- Do NOT generate explanations.
- After classifying all papers:
  - If all papers belong to the same domain (excluding "Other"), generate one concise confirmation question for that domain.  
  - If multiple domains are present, determine the majority domain and generate one concise confirmation question for that majority domain.  
  - If there is no clear majority domain, default to "Single-Atom Catalysts" and generate a confirmation question accordingly. 
  
OUTPUT FORMAT (valid JSON only):
{{
  "classifyed_papers": [
    {{
      "paper_id": "<paper_id>",
      "classification": "Single-Atom Catalysts" | "Photovoltaic Tandem Devices" | "Biopolymer Materials" | "Other"
    }},
    {{}
      "paper_id": "<paper_id>",
      "classification": "..."
    }},
    ...
  ],
  "question": "<a concise question asking if investigation should proceed in the chosen domain>"
}}
"""

clarify_with_user_and_paper="""You are an intelligent message–document alignment agent. 
Your task is to process a user message and a scientific paper (title + abstract), then perform three steps:

<Messages>
{messages}
</Messages>

<Papers>
{papers}
</Papers>

Categories:
- Single-Atom Catalysts
- Photovoltaic Tandem Devices
- Biopolymer Materials
- Other

1. **Identify the intent of the user message**  
   - Analyze the user’s message and summarize the research intent.  
   - Classify the message into one of the categories.   

2. **Classify the domain of each paper**  
   - For every paper, read the title and abstract.  
   - Classify the paper’s domain into one of the same four categories.   

3. **Generate a single clarification question**  
   - Determine the majority domain among all classified papers (excluding "Other").  
   - If the majority paper domain is the same as the message domain:  
     - Generate a concise confirmation question asking if the investigation should proceed in that domain.  
   - If the majority paper domain differs from the message domain:  
     - Generate a concise clarification question asking the user to **revise their message to align with the majority paper domain**.  
     - Example: “메시지는 단일 원자 촉매에 관한 것이지만, 첨부된 논문들은 바이오폴리머 소재에 관한 것입니다. 논문 도메인(바이오폴리머 소재)에 맞게 질문을 다시 해주시겠습니까?”  
        
OUTPUT FORMAT (valid JSON only)
{{
  "query_intent": "string",  
  "query_domain": "Single-Atom Catalysts" | "Photovoltaic Tandem Devices" | "Biopolymer Materials" | "Other",  
  "classifyed_papers": [
    {{
      "paper_id": "<paper_id>",
      "classification": "Single-Atom Catalysts" | "Photovoltaic Tandem Devices" | "Biopolymer Materials" | "Other"
    }},
    {{
      "paper_id": "<paper_id>",
      "classification": "..."
    }},
    ...
  ],  
  "question": "<a concise question asking if investigation should proceed in the chosen domain>"
}}    
"""

transform_messages_into_research_topic_prompt = """You will be given a set of messages that have been exchanged so far between yourself and the user. 
Your job is to translate these messages into a more detailed and concrete research question that will be used to guide the research.

The messages that have been exchanged so far between yourself and the user are:
<Messages>
{{ messages }}
</Messages>

{% if papers %}
The following papers were provided by the user as relevant to their research topic.
<Papers>
{{ papers }}
</Papers>
{% endif %}

You will return a single research question that will be used to guide the research.

Guidelines:
1. Maximize Specificity and Detail
- Include preferences and explicitly list key attributes or dimensions to consider.
- It is important that all details from the user are included in the instructions.

2. Handle Unstated Dimensions Carefully
- When research quality requires considering additional dimensions that the user hasn't specified, acknowledge them as open considerations rather than assumed preferences.
- Example: Instead of assuming "budget-friendly options," say "consider all price ranges unless cost constraints are specified."
- Only mention dimensions that are genuinely necessary for comprehensive research in that domain.

3. Avoid Unwarranted Assumptions
- Never invent specific user preferences, constraints, or requirements that weren't stated.
- If the user hasn't provided a particular detail, explicitly note this lack of specification.
- Guide the researcher to treat unspecified aspects as flexible rather than making assumptions.

4. Distinguish Between Research Scope and User Preferences
- Research scope: What topics/dimensions should be investigated (can be broader than user's explicit mentions)
- User preferences: Specific constraints, requirements, or preferences (must only include what user stated)

5. Use the First Person
- Phrase the request from the perspective of the user.
"""

evaluate_criteria_prompt = """You are a materials-science evaluation planner.
Your task is to infer quantitative weights (w_i) for six evaluation criteria 
based on the user's research goal, priorities, and trade-offs.

Each weight must be between 0 and 1, and all six weights must sum to exactly 1.0.

Evaluation Criteria:
1. Activity index (ΔG_H*, HER overpotential, TOF)
2. Stability (long-term CA/CP, leaching rate, bond retention)
3. Synthesis / support compatibility & scalability
4. Cost & abundance (economic feasibility)
5. Strength of literature evidence (reproducibility / meta-analysis score)
6. ML–literature agreement

---

research_brief:
{{ research_brief }}

{% if user_feedback %}
user_feedback:
{{ user_feedback }}
{% endif %}
---

You must:
1. Parse the user's intent and identify which factors are emphasized or deprioritized.
2. Assign higher weights to emphasized criteria and lower to neglected ones.
3. Justify each weight assignment concisely (1–2 lines).
4. Ensure Σw_i = 1.0 exactly.
5. Generate a polite confirmation sentence asking if the user wants to proceed with these weights.
   The tone should be concise and natural (e.g., "이 설정은 안정성과 재현성에 중점을 둡니다. 이 가중치를 적용할까요?").
6. If the user's query is ambiguous, use the default baseline:
   { "activity": 0.35, "stability": 0.25, "synthesis": 0.15, "cost": 0.10, "evidence": 0.10, "ml_lit_agree": 0.05 }.

Output Format (strict JSON):
{
  "weight": {
    "activity": float,
    "stability": float,
    "synthesis": float,
    "cost": float,
    "evidence": float,
    "ml_lit_agree": float
  },
  "rationale": {
    "activity": "string",
    "stability": "string",
    "synthesis": "string",
    "cost": "string",
    "evidence": "string",
    "ml_lit_agree": "string"
  },
  "feedback_question": "a concise, user-facing confirmation sentence asking whether to proceed with these weights"
}
"""

feedback_intent_prompt = """Your task:
Classify whether the user accepts the criteria as-is, or requests revisions.

### Decision Labels
- **"approve"** → The user agrees with or is satisfied with the evaluation criteria.  
  Typical signals: 긍정 표현, 수락, "좋아요", "그대로 진행", "괜찮아요", "문제 없습니다", "좋습니다", "이대로 해주세요"
- **"revise"** → The user rejects, disagrees, or asks for changes or clarifications.  
  Typical signals: 부정, 수정 요청, "수정해 주세요", "다시 만들어 주세요", "이 기준은 이상해요", "가중치 바꾸고 싶어요", "더 강조했으면 좋겠어요"

---

### Input
User feedback:
{user_feedback}

---

### Output Format (strict JSON only)
{{
  "intent": "approve" | "revise"
}}
"""

final_anwser_prompt = """You are the Final Answer Agent.

Your task is to generate the best possible answer to the user's query using
the following structured information:

1) <QUERY>             → 사용자가 입력한 질문
2) <SEARCH_RESULTS>    → 데이터베이스/문헌 검색 결과
3) <HYPOTHESIS>        → 가설 생성 에이전트가 제안한 가설
4) <DEBATE_SUMMARY>    → 토론 에이전트의 찬반·판정 요약

Follow these rules:
- All statements must be consistent with SEARCH_RESULTS.
- If HYPOTHESIS conflicts with the data, explicitly point out the conflict.
- If DEBATE_SUMMARY is inconclusive, propose next steps.
- Output must be concise, structured, and domain-expert level.

<QUERY>
{{ query }}
</QUERY>

{% if search_result %}
<SEARCH_RESULTS>
{% for s in search_result %}
<SEARCH_RESULT index="{{ loop.index }}">
{{ s }}
</SEARCH_RESULT>
{% endfor %}
</SEARCH_RESULTS>
{% endif %}

{% if hypothesis %}
<HYPOTHESIS>
{% for h in hypothesis %}
<HYPOTHESIS_ITEM index="{{ loop.index }}">
{{ h }}
</HYPOTHESIS_ITEM>
{% endfor %}
</HYPOTHESIS>
{% endif %}

{% if debate_summary %}
<DEBATE_SUMMARY>
  <ISSUE>{{ debate_summary.issue }}</ISSUE>
  <PROPONENT_SUMMARY>{{ debate_summary.proponent_summary }}</PROPONENT_SUMMARY>
  <OPPONENT_SUMMARY>{{ debate_summary.opponent_summary }}</OPPONENT_SUMMARY>
  <AGREEMENT_STATUS>{{ debate_summary.agreement_status }}</AGREEMENT_STATUS>
</DEBATE_SUMMARY>
{% endif %}
"""