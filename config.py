"""Configuration for OCR benchmark evaluation system."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str
    model_id: str
    backend: str  # "vllm", "mineru", "glmocr_pipeline", "paddleocr_pipeline"
    vllm_args: list[str] = field(default_factory=list)
    port: int = 8000


MODELS: dict[str, ModelConfig] = {
    "glm-ocr": ModelConfig(
        name="GLM-OCR",
        model_id="zai-org/GLM-OCR",
        backend="vllm",
        vllm_args=[
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb", "0",
            "--max-model-len", "16384",
        ],
    ),
    "paddleocr-vl": ModelConfig(
        name="PaddleOCR-VL",
        model_id="PaddlePaddle/PaddleOCR-VL",
        backend="vllm",
        vllm_args=[
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb", "0",
            "--max-num-batched-tokens", "16384",
        ],
    ),
    "deepseek-ocr2": ModelConfig(
        name="DeepSeek-OCR2",
        model_id="deepseek-ai/DeepSeek-OCR-2",
        backend="vllm",
        vllm_args=[
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb", "0",
            "--max-model-len", "8192",
        ],
    ),
    "mineru": ModelConfig(
        name="MinerU-2.5",
        model_id="mineru",
        backend="mineru",
    ),
    "glm-ocr-pipeline": ModelConfig(
        name="GLM-OCR-Pipeline",
        model_id="zai-org/GLM-OCR",
        backend="glmocr_pipeline",
        vllm_args=[
            "--trust-remote-code",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb", "0",
            "--max-model-len", "16384",
            "--served-model-name", "glm-ocr",
            "--gpu-memory-utilization", "0.80",
        ],
    ),
    "paddleocr-vl-pipeline": ModelConfig(
        name="PaddleOCR-VL-Pipeline",
        model_id="PaddlePaddle/PaddleOCR-VL",
        backend="paddleocr_pipeline",
    ),
}


@dataclass
class BenchmarkConfig:
    name: str
    dataset_id: str
    split: str
    metric_type: str
    prompt_key: str
    max_samples: int | None  # None = full dataset


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "omnidocbench": BenchmarkConfig(
        name="OmniDocBench",
        dataset_id="opendatalab/OmniDocBench",
        split="train",
        metric_type="document_parse",
        prompt_key="document_parse",
        max_samples=None,  # 전수 (1,358)
    ),
    "upstage_dp_bench": BenchmarkConfig(
        name="Upstage DP-Bench",
        dataset_id="upstage/dp-bench",
        split="test",
        metric_type="document_parse_dp",
        prompt_key="document_parse_dp",
        max_samples=None,  # 전수 (~200)
    ),
    "ocrbench": BenchmarkConfig(
        name="OCRBench",
        dataset_id="echo840/OCRBench",
        split="test",
        metric_type="text_recognition",
        prompt_key="text_recognition",
        max_samples=None,  # 전수 (1,000)
    ),
    "unimernet": BenchmarkConfig(
        name="UniMERNet",
        dataset_id="deepcopy/UniMER",
        split="test",
        metric_type="formula_recognition",
        prompt_key="formula_recognition",
        max_samples=200,
    ),
    "pubtabnet": BenchmarkConfig(
        name="PubTabNet",
        dataset_id="apoidea/pubtabnet-html",
        split="validation",  # test split 없음
        metric_type="table_recognition",
        prompt_key="table_to_html",
        max_samples=200,
    ),
    "teds_test": BenchmarkConfig(
        name="TEDS_TEST",
        dataset_id="apoidea/pubtabnet-html",
        split="validation",  # test split 없음
        metric_type="table_recognition",
        prompt_key="table_to_html",
        max_samples=200,
    ),
    "nanonets_kie": BenchmarkConfig(
        name="Nanonets-KIE",
        dataset_id="nanonets/key_information_extraction",
        split="test",
        metric_type="kie_extraction",
        prompt_key="kie_extraction",
        max_samples=None,  # 전수
    ),
    "handwritten_forms": BenchmarkConfig(
        name="Handwritten-Forms",
        dataset_id="Teklia/IAM-line",
        split="test",
        metric_type="handwritten",
        prompt_key="handwritten",
        max_samples=200,
    ),
}

RESULTS_DIR = Path("/home/ubuntu/ocr_test/results")
DATA_CACHE_DIR = Path("/home/ubuntu/ocr_test/data_cache")
PREPARED_DIR = Path("/home/ubuntu/ocr_test/prepared_datasets")

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_DIR.mkdir(parents=True, exist_ok=True)
