import deepchem as dc
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pymatgen.core import Element
import joblib
import os
import re
import warnings
from pymatgen.core import Composition

# 특수한 원소들의 경울 특정 물성치값이 없을수 있다는 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)
base_dir = os.path.dirname(os.path.abspath(__file__))

mcp = FastMCP("perovskite_tools")

def load_dict():
    df = pd.read_csv(os.path.join(base_dir, "perovskite_mapping.csv"))
    mapping = {row.Token: row.Converted for row in df.itertuples()}
    return mapping

def preprocess(formula: str):
    """
    사전에 있는 약어를 우선순위(긴 것 먼저)로 안전하게 치환.
    - 주의: 약어가 다른 문자열 일부로 포함될 수 있어, 길이 내림차순으로 처리
    """
    if not formula:
        return formula

    abbrev_map = load_dict()

    # 긴 약어부터 치환(예: BA와 BAI가 같이 있을 때 안전)
    keys = sorted(abbrev_map.keys(), key=len, reverse=True)

    out = formula
    for ab in keys:
        exp = abbrev_map[ab]

        # 대소문자 구분. 약어는 보통 대문자 중심이라 그대로 처리.
        # "알파벳 덩어리 내에서만" 치환되도록 정규식 사용:
        # - 앞뒤가 알파벳이 아니거나 문자열 경계일 때만 치환
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(ab)}(?![A-Za-z])")
        out = pattern.sub(exp, out)

        # MAPbI3처럼 붙어있는 경우는 위 패턴이 안 잡힐 수 있어서,
        # 알파벳 덩어리 내부 치환도 추가로 수행(단, 원소기호와 충돌 최소화 위해 약어 사전 기반으로만)
        pattern_inside = re.compile(rf"{re.escape(ab)}")
        out = pattern_inside.sub(exp, out)

    return out

@mcp.tool()
def element_fingerprint(formula: str):
    """
    화학식을 입력받아 원소의 물리화학적 특성(Element Property Fingerprint)을 기반으로 재료의 물성을 예측합니다.
    
    이 도구는 각 원소의 특성(원자 번호, 전기음성도, 원자 반지름 등)의 통계적 요약값(평균, 편차 등)을 
    피처로 추출하며, 학습된 Random Forest 모델을 사용하여 결과를 반환합니다. 
    상대적으로 데이터 세트가 적을 때도 견고한 예측 성능을 보입니다.

    Args:
        formula (str): 예측하고자 하는 화학식 (예: 'CsPbI3', 'Cs0.5PbBrI2').
                       내부적으로 전처리를 통해 정규화된 형식으로 변환됩니다.

    Returns:
        float: Random Forest 모델에 의해 예측된 해당 재료의 물성 값 (예: 형성 에너지 또는 밴드갭).
    """
    # fomula 전처리
    processed_formula = preprocess(formula)
    featurizer = dc.feat.ElementPropertyFingerprint()
    features = featurizer.featurize([processed_formula])

    if features.size == 0:
        print(features)

    # model 로드
    rf_model = joblib.load(os.path.join(base_dir, "./model/material_rf.joblib"))
    yte = rf_model.predict(features)

    return yte[0]

@mcp.tool()
def element_net(formula: str):
    """
    화학식을 입력받아 ElemNet 기반의 원소 비율 벡터를 생성하고, 이를 통해 재료의 물성을 예측합니다.
    
    이 도구는 DeepChem의 ElemNetFeaturizer를 사용하여 주기율표 상의 원소 존재 여부와 비율을 
    벡터화합니다. 이후 학습된 Kernel Ridge Regression(KRR) 모델을 사용하여 비선형적인 
    관계까지 고려한 물성 예측치를 제공합니다.

    Args:
        formula (str): 예측하고자 하는 화학식 (예: 'CsPbI3').
                       화학 원소와 그 수량 비율이 명확해야 정확한 피처 생성이 가능합니다.

    Returns:
        float: KRR 모델에 의해 예측된 해당 재료의 물성 값.
    """
    # fomula 전처리
    processed_formula = preprocess(formula)
    featurizer = dc.feat.ElemNetFeaturizer()
    features = featurizer.featurize([processed_formula])

    if features.size == 0:
        print(features)

    # model 로드
    krr_model = joblib.load(os.path.join(base_dir, "./model/material_graph_krr.joblib"))
    yte = krr_model.predict(features)

    return yte[0]

if __name__ == "__main__":
    mcp.run()