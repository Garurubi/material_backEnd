
from enum import Enum
from typing import List, Optional, Union, Type, Literal, Annotated, Dict, Any
from pydantic import BaseModel, Field, ValidationError, field_validator
class OperationTypeEnum(str, Enum):
    starting = "StartingSynthesis"
    mixing = "MixingOperation"
    shaping = "ShapingOperation"
    drying = "DryingOperation"
    heating = "HeatingOperation"
    quenching = "QuenchingOperation"
    post_treatment = "PostTreatmentOperation"

class ActiveMetal(BaseModel):
    element: str
    oxidationState: Optional[str] = None
    content: Optional[str] = None
    coordinationEnvironment: Optional[str] = None

class CatalystComposition(BaseModel):
    activeMetals: List[ActiveMetal]
    supportMaterial: Optional[str] = None

class Catalyst(BaseModel):
    name: str
    type: Optional[str] = None
    composition: Optional[CatalystComposition] = None

class Precursor(BaseModel):
    name: str
    formula: Optional[str] = None

class HeatingValue(BaseModel):
    max_value: float
    min_value: float
    units: str
    value: List[float]

class HeatingConditions(BaseModel):
    heating_temperature: Optional[List[HeatingValue]] = None
    heating_time: Optional[List[HeatingValue]] = None
    heating_atmosphere: Optional[List[str]] = None
    heating_rate: Optional[str] = None
    
    @field_validator('heating_atmosphere', mode='before')
    @classmethod
    def filter_none_values(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            # 리스트에서 None 값들을 필터링하고, 빈 리스트가 되면 None 반환
            filtered = [item for item in v if item is not None]
            return filtered if filtered else None
        return v

class MixingConditions(BaseModel):
    mixing_device: Optional[str] = None
    mixing_media: Optional[str] = None
    mixing_duration: Optional[str] = None

class HeatingStep(BaseModel):
    stepNumber: int
    operation: str
    type: Literal[OperationTypeEnum.heating] = OperationTypeEnum.heating
    conditions: HeatingConditions

class MixingStep(BaseModel):
    stepNumber: int
    operation: str
    type: Literal[OperationTypeEnum.mixing] = OperationTypeEnum.mixing
    conditions: MixingConditions

class OtherStep(BaseModel):
    stepNumber: int
    operation: str
    type: Literal[
        OperationTypeEnum.starting,
        OperationTypeEnum.shaping,
        OperationTypeEnum.drying,
        OperationTypeEnum.quenching,
        OperationTypeEnum.post_treatment,
    ]

SynthesisStep = Union[HeatingStep, MixingStep, OtherStep]

class SynthesisProcess(BaseModel):
    overallMethod: Optional[str] = None
    steps: List[Annotated[SynthesisStep, Field(discriminator="type")]]
    precursors: List[Precursor]

class Electrolyte(BaseModel):
    composition: Optional[str] = None
    concentration: Optional[str] = None
    pH: Optional[Union[float, str]] = None
    type: Optional[str] = None

class PerformanceConditions(BaseModel):
    electrolyte: Electrolyte
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None
    scanRate: Optional[str] = None
    additionalIons: Optional[str] = None
    pressure: Optional[str] = None

class OverpotentialMetric(BaseModel):
    value: Union[float, str]
    unit: str
    currentDensity: str

class OnsetPotentialMetric(BaseModel):
    value: float
    unit: str
    currentDensity: str

class TafelSlopeMetric(BaseModel):
    value: float
    unit: str

class TOFMetric(BaseModel):
    value: float
    unit: str
    conditions: str

class MassActivityMetric(BaseModel):
    value: float
    unit: str

class StabilityMetric(BaseModel):
    testType: str
    currentDensity: str
    duration: str
    degradation: str

class SelectivityMetric(BaseModel):
    pathway: str
    H2O2generation: Optional[str] = None
    electronTransferNumber: Optional[str] = None

class PerformanceMetrics(BaseModel):
    overpotential: Optional[OverpotentialMetric] = None
    onsetPotential: Optional[OnsetPotentialMetric] = None
    tafelSlope: Optional[TafelSlopeMetric] = None
    turnoverFrequency: Optional[TOFMetric] = None
    massActivity: Optional[MassActivityMetric] = None
    stability: Optional[StabilityMetric] = None
    faradaicEfficiency: Optional[str] = None
    selectivity: Optional[SelectivityMetric] = None
    additionalNotes: Optional[str] = None

class ElectrochemicalPerformance(BaseModel):
    reaction: str
    conditions: PerformanceConditions
    metrics: PerformanceMetrics

class Experiment(BaseModel):
    catalyst: Catalyst
    synthesisProcess: SynthesisProcess
    electrochemicalPerformance: ElectrochemicalPerformance

class extracted_data(BaseModel):
    experiments: List[Experiment]
class catalyst(BaseModel) :
    _id : str
    name : str
    extracted_data : extracted_data