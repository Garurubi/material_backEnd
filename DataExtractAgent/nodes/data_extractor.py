
from ..state import extractState
from ..schemas.sacs.electro_chemical import *
from ..state import Response
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from typing import List, Optional, Union 
from dotenv import load_dotenv
# load_dotenv()
# --- OpenAI 클라이언트 전역 변수 설정 ---
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Error: OPENAI_API_KEY environment variable not set. Please set the key to run the script.")
client = OpenAI()
max_workers = 3

# 멀티 쓰레드를 통한 병렬 처리
def data_extract(state: extractState) -> extractState:
    responses =[]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(data_extract_from_text, pdf_text , templete_prompt): (pdf_text , templete_prompt) for pdf_text , templete_prompt in zip(state["pdf_text"] , state["templete_prompt_list"])}
        for i, future in enumerate(as_completed(future_to_file), 1):
            try:
                result= future.result()
                responses.append(result)
            except Exception as e:
                print(f"❌ 예외 발생: {e}")
                responses.append({})
    return {"response":responses}


def data_extract_from_text(pdf_text: str ,templete_text:str) -> Response:
    # 받아온 텍스트와 이것으로 처리
    # 어떤것으로 처리하는지? 확인해보면
    
    previous_feedback = None
    max_iterations=2
    iteration_history=[]
    extraction_model: str = "gpt-5"
    supervisor_model: str = "gpt-5-nano"
    min_confidence_threshold = 0.6
 

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        
        # 프롬프트 생성 (클래스 코드 , 몇 회차인지, 이전 피드백)
        prompt = create_extraction_prompt(
            templete_text, 
            iteration, 
            previous_feedback
        )
        
        # LLM 호출 - (pdf 경로, 생성한 프롬프트)
        print("Calling extraction LLM...")
        #첫번째 추출 실행
        extracted_data = call_extraction_llm(pdf_text, prompt,extraction_model)
        
        
        if not extracted_data:
            print(f"❌ Extraction failed at iteration {iteration}")
            continue
        
        # Supervisor 검증
        print("Calling supervisor for validation...") 
        supervisor_result = call_supervisor_llm(extracted_data,supervisor_model) 

        if not supervisor_result:
            print(f"❌ Supervisor validation failed at iteration {iteration}")
            continue
        
        # 기록 저장
        iteration_history.append({
            "iteration": iteration,
            "extracted_data": extracted_data,
            "supervisor_result": supervisor_result,
            "confidence_score": supervisor_result.confidence_score
        })
        
        print(f"Supervisor confidence: {supervisor_result.confidence_score:.2f}")
        print(f"Issues found: {len(supervisor_result.issues)}")
        
        # 검증 통과 시 JSON 응답으로 진행
        if supervisor_result.is_valid and supervisor_result.confidence_score >=  min_confidence_threshold:
            print(f"✅ Supervisor validation passed with confidence {supervisor_result.confidence_score:.2f}!")
            break
        elif supervisor_result.is_valid and supervisor_result.confidence_score <  min_confidence_threshold:
            print(f"⚠️ Validation passed but confidence too low ({supervisor_result.confidence_score:.2f} < {min_confidence_threshold})")
            previous_feedback = f"Previous attempt had low confidence ({supervisor_result.confidence_score:.2f}). {supervisor_result.feedback}"
        else:
            print(f"❌ Supervisor found issues: {supervisor_result.issues}")
            previous_feedback = supervisor_result.feedback
            
        if iteration == max_iterations:
            print("⚠️ Max iterations reached.")
            # 최소 임계값을 만족하는 결과가 있는지 확인
            valid_results = [h for h in  iteration_history if h["confidence_score"] >=  min_confidence_threshold]
            
            if valid_results:
                # 임계값을 만족하는 결과 중 가장 높은 점수 선택
                best_result = max(valid_results, key=lambda x: x["confidence_score"])
                extracted_data = best_result["extracted_data"]
                print(f"Using best valid result with confidence {best_result['confidence_score']:.2f}")
            else:
                # 임계값을 만족하는 결과가 없음
                print(f"❌ No results met minimum confidence threshold ({ min_confidence_threshold})")


                ## 20251001 - 추출 결과 threshold를 못넘기는 경우는 어떻게 할 것 인지?
                # 그나마 나은 결과 사용
                extracted_data = max(iteration_history,key=lambda x: x["confidence_score"])["extracted_data"]
                # 나쁜 결과는 모두 실패 판정
                # return {"status":"failure","error_message":f"No results met minimum confidence threshold  : {e}"}
            break
    
    # JSON 응답 처리 - 신뢰도 체크 추가 + 빈 리스트 검증
    if extracted_data is None or not extracted_data:
        print(f"❌ Extraction failed: No results met minimum confidence threshold ({ min_confidence_threshold})")
        return {"status":"failure","error_message":f"Extraction failed : {e}"} 
    try:
        parsed_data = json.loads(extracted_data)
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        return {"status":"failure","error_message":f"JSON Decode Error {e}, \nextracted_data : {extracted_data}"}
    
    # Pydantic 검증
    try:
        validated_output = FinalOutput(**parsed_data)
        print("✅ Final Pydantic validation successful!")
        return {"status":"success" , "response":validated_output}
    except ValidationError as e:
        print(f"❌ Final Pydantic Validation Error: {e}")
        return {"status":"failure","error_message":f"pydantic validation error {e}"}




from dataclasses import dataclass

@dataclass
class SupervisorResult:
    is_valid: bool
    feedback: str
    confidence_score: float
    issues: List[str]

class SupervisorResponse(BaseModel):
    is_valid: bool
    feedback: str
    confidence_score: float
    issues: List[str]
    suggestions: Optional[Union[str, List[str]]] = None

def create_extraction_prompt(  pydantic_models_code: str, iteration: int = 1, 
                            previous_feedback: str = None) -> str:
        """데이터 추출을 위한 프롬프트 생성"""
        base_prompt = f"""
You are an expert in electrochemical catalysis tasked with extracting structured data from scientific papers on single-atom catalysts.

**OBJECTIVE**: Extract data from the attached PDF and output a single JSON object that strictly validates against the provided enhanced Pydantic schema.

**OUTPUT STRUCTURE (CRITICAL)**:
- Return **`experiments`**: a list where **each element is one experimental set** consisting of:
  - `catalyst`
  - `synthesisProcess` 
  - `electrochemicalPerformance`
- If the paper reports multiple catalysts, or multiple experimental variants, **return multiple items** in `experiments`. Do **not** merge distinct catalysts/experiments.

**EXTRACTION PRINCIPLES:**
1. **Fidelity First**: Extract ONLY explicitly stated information. Never infer, interpret, or fill gaps with assumptions.
2. **Verbatim Accuracy and Operation Specificity**: 
   - Chemical names: Preserve exact notation.
   - Numerical values: Include all significant figures and units exactly as written.
   - `operation` field: Extract the most specific technical term for the action.
3. **Detailed Parameter Inclusion (Enhanced)**:
   - **Active Metal Oxidation State**: Extract oxidation states if and only if explicitly provided.
   - **Support Material**: Extract if explicitly provided.
   - **Heating Rate and Mixing Duration**: Precisely extract if explicitly stated in the synthesis steps.
   - **Electrolyte Concentration**: Precisely extract if explicitly stated.
4. **Additional Notes**: 
   - Include significant notes or specific remarks explicitly highlighted by the authors regarding performance evaluation or synthesis details.
5. **Null Handling Protocol**:
   - Use `null` for any missing Optional fields. Do not assume default or common values.
   - **NEVER put `null` values inside arrays/lists**. If a list field is missing, use `null` for the entire field, not `[null]` or `["value", null]`.
   - For example: Use `"heating_atmosphere": null` NOT `"heating_atmosphere": [null]` or `"heating_atmosphere": ["air", null]`.
6. **Schema-Driven Field Inclusion**:
   - The schema uses discriminated unions for synthesis steps.
   - `conditions` **must exist only** for `"HeatingOperation"` and `"MixingOperation"`.
   - For other types, **do NOT** include a `conditions` field.
7. **Reaction Classification Constraint**:
   - The field `electrochemicalPerformance.reaction` MUST be classified into exactly one of the following categories:
     - `"CO2RR"` (Carbon dioxide reduction reaction)
     - `"NO3RR"` (Nitrate reduction reaction)
     - `"HER"` (Hydrogen evolution reaction)
     - `"OER"` (Oxygen evolution reaction)
     - `"Other"` (any other electrochemical reaction explicitly reported, not fitting the above categories)
   - Extract verbatim terms from the paper, but map them into one of the above categories.
   - If the reaction cannot be clearly assigned, classify it as `"Other"`.
**OPERATION TYPE CLASSIFICATION** (use these EXACT strings):
- `"StartingSynthesis"` – initial material preparation, dissolution, precipitation reactions
- `"MixingOperation"` – stirring, sonication, grinding, dispersing
- `"ShapingOperation"` – molding, pressing, forming
- `"DryingOperation"` – evaporation, vacuum drying, freeze-drying
- `"HeatingOperation"` – pyrolysis, calcination, annealing, carbonization
- `"QuenchingOperation"` – rapid cooling, thermal shock
- `"PostTreatmentOperation"` – acid/base leaching, washing, purification, activation

**Enhanced Pydantic Models Schema**:
```python
{pydantic_models_code}
```
"""
        # 첫번째가 아니고 피드백이 있다면
        if iteration > 1 and previous_feedback:
            # 사족추가
            feedback_section = f"""
        
**IMPORTANT - FEEDBACK FROM PREVIOUS ATTEMPT (Iteration {iteration}):**
{previous_feedback}

**Please address the above issues and improve the extraction accordingly.**
"""
            base_prompt += feedback_section

        return base_prompt




def call_extraction_llm(  pdf_text: str, prompt: str,extraction_model:str) -> Optional[str]:
    """GPT-5를 사용한 추출 LLM 호출 - Docling 파싱 텍스트를 프롬프트에 포함"""
    try:
        
        
        # 텍스트 길이 제한 (토큰 보호)
        max_chars = 100000
        if len(pdf_text) > max_chars:
            print(f"텍스트가 너무 깁니다 ({len(pdf_text)} chars). 처음 {max_chars} 문자로 제한합니다.")
            pdf_text = pdf_text[:max_chars] + "\n\n[텍스트가 잘렸습니다...]"
        
        # 파싱된 텍스트를 프롬프트에 포함
        full_input = f"""Please analyze the following PDF content and follow the instructions:

PDF Content:
{pdf_text}

Instructions:
{prompt}"""
        
        result = client.responses.create(
            model=extraction_model,
            input=full_input,
            reasoning={"effort": "low"},
            text={"verbosity": "low"}
        ) 
        print(f"GPT-5 분석 성공. 응답 길이: {len(result.output_text)} 문자") 
        return result.output_text
        
    except Exception as e:
        print(f"Error in extraction LLM call: {e}")
        return None


def call_supervisor_llm(  extracted_data: str,supervisor_model:str) -> Optional[SupervisorResult]:
    """GPT-5-nano를 사용한 Supervisor LLM 호출"""
    try:
        supervisor_prompt = create_supervisor_prompt(extracted_data)
        
        # GPT-5-nano는 responses API를 사용 (GPT-5와 동일한 방식)
        result = client.responses.create(
            model=supervisor_model,
            input=supervisor_prompt,
            reasoning={"effort": "low"},
            text={"verbosity": "low"}
        ) 


        response_data = json.loads(result.output_text)
        supervisor_response = SupervisorResponse(**response_data)
        
        return SupervisorResult(
            is_valid=supervisor_response.is_valid,
            feedback=supervisor_response.feedback,
            confidence_score=supervisor_response.confidence_score,
            issues=supervisor_response.issues
        )
        
    except Exception as e:
        print(f"Error in supervisor LLM call: {e}")
        return SupervisorResult(
            is_valid=False,
            feedback=f"Supervisor error: {e}",
            confidence_score=0.0,
            issues=[f"Supervisor evaluation failed: {e}"]
        )



def create_supervisor_prompt( extracted_data: str) -> str:
    """Supervisor가 추출된 데이터를 검증하기 위한 프롬프트"""
    return f"""
You are a quality assurance supervisor for scientific data extraction from electrochemical catalysis papers.

**TASK**: Review the extracted JSON data below and determine if it meets high quality standards for scientific accuracy and completeness.

**EVALUATION CRITERIA:**
1. **Schema Compliance**: Does the data follow the correct JSON structure?
2. **Data Completeness**: Are key experimental details present (catalyst composition, synthesis steps, performance metrics)?
3. **Scientific Accuracy**: Do the chemical names, units, and values appear scientifically reasonable?
4. **Consistency**: Are there any internal contradictions or inconsistencies?
5. **Missing Critical Information**: Are there obvious gaps in essential data that should be present?

**TOLERANCE & SCORING POLICY (apply leniently where safe):**
- Minor omissions in optional fields (e.g., `additionalNotes`, optional metrics) are acceptable.
- Small unit-format differences or rounding within typical lab precision are acceptable.
- Prefer constructive feedback over outright rejection when the structure is valid and the majority of core content is present.
- Set `is_valid: true` if the JSON is schema-valid and covers the core fields, even if some optional details are missing; reflect uncertainty in `confidence_score`.

**EXTRACTED DATA TO REVIEW:**
```json
{extracted_data}
```

**OUTPUT FORMAT**: Provide your assessment as a JSON object with the following structure:
```json
{{
"is_valid": true/false,
"feedback": "Detailed feedback on what can be improved (be specific but concise)",
"confidence_score": 0.0-1.0,
"issues": ["list", "of", "specific", "issues"],
"suggestions": "Specific, actionable suggestions to improve extraction (optional)"
}}

**DECISION GUIDELINES:**
- Use a threshold notionally around 0.6 for high confidence, but allow `is_valid: true` below this if issues are minor and easily fixable.
- Set `is_valid: true` when the data is of generally good quality with only minor issues; otherwise `false`.
- `confidence_score` should reflect how confident you are in the extraction quality (0.0 = very poor, 1.0 = excellent)
"""
