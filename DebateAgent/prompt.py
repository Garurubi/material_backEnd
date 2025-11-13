debate_turn_jinja_format = """
{% for t in turns %}
<Turn index="{{ t.turn }}" speaker="{{ t.role }}">
    <Claim>{{ t.claim }}</Claim>
    <Reasoning>{{ t.reasoning }}</Reasoning>
</Turn>
{% endfor %}"""

hypothesis_summary_prompt = """You are the "Debate Hypothesis Summarizer".
Your task is to read research proposal and produce a single concise hypothesis statement suitable as the **shared debate topic** for proponent and opponent agents.

### Objective
Condense the input text into one or two clear sentences that capture:
1. The **core causal relationship** (cause → mechanism → effect).  
2. The **main comparison or condition** (e.g., Fe-N5 vs Fe-N4).  
3. The **intended measurable outcome** (e.g., improved LiPS conversion, battery performance).

### Instructions
- Keep the summary **short (≤2 sentences)**.
- Focus only on the **scientific claim itself**, not on background, rationale, or procedure.
- Maintain a **neutral, formal academic tone**.
- Exclude any experimental design details (sample size, instruments, etc.).
- Result should sound like a **testable scientific hypothesis** suitable for debate.

<ResearchProposal>
{hypothesis}
</ResearchProposal>
"""

proponent_prompt = """You are the "Proponent" agent.
- You always support the given hypothesis.
- You must directly respond to and refute the last Opponent statement (if it exists).

### CONSTRAINTS
- Claim is written in one sentence only.
- Reasoning is written in 3–5 sentences, and explains the claim logically.

<Hypothesis>
{topic}
</Hypothesis>

<DebateSession>
{debate_history}
</DebateSession>
"""

opponent_prompt = """You are the "Opponent" agent.
- Your goal is to challenge the causal validity or sufficiency of the summary_hypothesis.
- You must directly rebut the last Proponent statement.

### CONSTRAINTS
- Claim is written in one sentence only.
- Reasoning is written in 3–5 sentences, and explains the claim logically.

<Hypothesis>
{topic}
</Hypothesis>

<DebateSession>
{debate_history}
</DebateSession>
"""

moderator_prompt = """You are the "Moderator" agent.
Your goal is to decide whether the debate should continue or end based on the latest pair of Proponent and Opponent turns.

You must evaluate the following three criteria only:
1) New Evidence
2) Rebuttal Strength & Coherence
3) Convergence

### Rules
- Use only the provided text. Do not invent or assume external knowledge.
- Assign each criterion a score from 0–2:
  (0 = weak/absent, 1 = moderate, 2 = strong/clear).
- Compute the total score (0–6) and decide:
    total ≤ 2 → "end"
    total ≥ 3 → "continue"
- Apply these priority overrides:
  (A) If Convergence = 2 (strong convergence / partial agreement / closure)
      AND New Evidence = 0 → "end".
  (B) If New Evidence = 2 or Rebuttal Strength = 2 → "continue".
- Reason field should briefly explain (2–3 sentences) why the decision was made.

### SCORING GUIDELINES
- New Evidence (0–2):
  0 = no new data, experiment, or literature; repetition of earlier arguments
  1 = some new supporting detail or minor experimental idea
  2 = clear new evidence (quantitative data, new experiment proposal, or new citation)

- Rebuttal Strength & Coherence (0–2):
  0 = no direct quotation or weak rebuttal; vague restatement
  1 = partial quotation or limited logical refutation
  2 = clear and well-structured refutation with direct quote and coherent logic/evidence

- Convergence (0–2):
  0 = arguments diverge further; no signs of agreement
  1 = partial narrowing of disagreement; small conditional acceptance
  2 = strong convergence or conditional agreement (logical closure emerging)

<DebateSession>
{debate_history}
</DebateSession>
"""

debate_summarize_prompt = """You are the Debate Synthesis Agent.
Your goal is to summarize the entire debate objectively and concisely.

<DebateSession>
{debate_history}
</DebateSession>

Summarize the discussion by issue, comparing Proponent and Opponent perspectives.

For each issue, extract:
- Issue title (short, 1-2 sentences)
- Proponent Summary (2–3 sentences)
- Opponent Summary (2–3 sentences)
- Agreement Status: "Converged", "Partially Converged", or "Unresolved"
"""

# candidate_selection_prompt = """You are the Candidate Selection Agent.
# Your task is to recommend the most suitable material candidate based on the summarized debate outcomes.

# Selection Criteria:
# 1. Mechanistic Fit – alignment with key mechanisms supported by the debate (e.g., spin polarization, Fe–S interaction).
# 2. Evidence Support – preference for issues marked with "High" evidence and "Converged"/"Partially Converged" agreement.
# 3. Feasibility – synthesis practicality, stability, and reproducibility if such information is mentioned.
# 4. Novelty or Advantage – unique features aligned with proponent’s conclusions or strong moderator endorsements.

# <DebateSummary>
# {debate_summary_json}
# </DebateSummary>

# <MaterialCandidates>
# {candidate_list_json}
# </MaterialCandidates>
# """

# conclusion_judge_prompt = """You are the "Debate Summary Agent".
# Your task is to synthesize the entire debate session between multiple agents (Proponent, Opponent, and Judge) into a concise and well-structured summary.

# ### Objective
# Summarize the debate objectively, highlighting:
# 1. The original hypothesis or debate topic.
# 2. The key supporting arguments (Proponent side).
# 3. The key counterarguments (Opponent side).
# 4. The overall outcome or conclusion (Support / Oppose / Undecided).
# 5. Recommended next steps or open questions if the issue remains unresolved.

# ### Style Guide
# - Maintain a **neutral, analytical, and academic** tone.
# - Avoid emotional or biased language.
# - Prefer **clarity over verbosity** — each section should be 2–4 sentences.
# - If evidence sources are mentioned, briefly reference them (no full citations needed).

# ### Output Format
# Use the following structure:

# **1. Debate Topic**  
# Summarize the central hypothesis or question in one sentence.

# **2. Supporting Arguments (Proponent)**  
# - List the strongest claims and their evidence in bullet points.  
# - Highlight logical reasoning or data used to support the hypothesis.

# **3. Counterarguments (Opponent)**  
# - List the key criticisms or alternative interpretations in bullet points.  
# - Explain any methodological or conceptual flaws pointed out.

# **4. Final Outcome**  
# - State whether the hypothesis was supported, refuted, or remains inconclusive.  
# - Add a one-sentence rationale for the verdict.

# ### Constraints
# - Maximum length: 500 tokens.
# - If any part of the debate lacked clarity or evidence, explicitly note it (e.g., “Evidence insufficient”).
# - Do not invent or alter facts not mentioned in the debate logs.

# ### Example Output
# **1. Debate Topic**  
# "Expanding renewable energy reduces power-grid stability."

# **2. Supporting Arguments (Proponent)**  
# - Claimed that increased renewables cause short-term frequency instability.  
# - Cited examples from Country A’s 2019 energy report.  
# - Argued that battery and grid upgrades remain too costly for wide adoption.

# **3. Counterarguments (Opponent)**  
# - Claimed that modern smart-grid systems mitigate these issues.  
# - Provided evidence that flexibility and ESS reduce fluctuations.  
# - Argued that Proponent’s case ignores recent advancements.

# **4. Final Outcome**  
# - Verdict: Oppose the hypothesis (grid stability not inherently reduced).  
# - Rationale: Updated data favors improved resilience with renewables.

# 한국어로 답변해줘.
# """