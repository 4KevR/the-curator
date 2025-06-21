#
#
#
# =============================  VARIABLES  ========================================================================

# random state for the test shuffling. Use None to disable shuffling (not recommended, may lead to autocorrelation)
random_state: int | None = 2308421

# if you only want to run a subset of the tests, set this.
subset_tests: slice | None = None  # slice(3, 15)

# options: 'local_llama', 'kit_llama', 'local_qwen8', 'local_qwen14'
llms_to_use: str = "local_llama"

# Where are the audio files? If asr should be skipped (only using text prompts), set to None.
audio_file_path: str | None = "../data/recording_data/fabian"

# options: 'local_whisper_medium', 'lecture_translator'
asrs_to_use: str = "local_whisper_medium"

default_temperature: float = 0.05
default_max_tokens: int = 2048

# ==================================================================================================================

import time  # noqa E402

script_start_time = time.time()

import logging  # noqa E402
import os  # noqa E402
import sys  # noqa E402
import warnings  # noqa E402
from datetime import datetime  # noqa E402
from zoneinfo import ZoneInfo  # noqa E402

import pandas as pd  # noqa E402
from dotenv import load_dotenv  # noqa E402

# must happen before project imports!
if os.path.exists("../../.env.db") and os.path.exists("../../.env"):
    base_path = "../../"
elif os.path.exists("../.env.db") and os.path.exists("../.env"):
    base_path = "../"
elif os.path.exists(".env") and os.path.exists(".env.db"):
    base_path = "./"
else:
    raise FileNotFoundError("No .env or .env.db found.")

sys.path.append(os.path.abspath(base_path))
load_dotenv(f"{base_path}.env")
load_dotenv(f"{base_path}.env.db")

logging.basicConfig(level=logging.WARN)

# Suppress FutureWarning from transformers
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

from src.backend.modules.asr.cloud_lecture_translator import CloudLectureTranslatorASR  # noqa E402
from src.backend.modules.asr.local_whisper_asr import LocalWhisperASR  # noqa E402
from src.backend.modules.evaluation.load_test_data.load_test_data import load_test_data  # noqa E402
from src.backend.modules.evaluation.run_tests.evaluation_pipeline import EvaluationPipeline  # noqa E402
from src.backend.modules.llm.kit_llm_req import KitLLMReq  # noqa E402
from src.backend.modules.llm.lm_studio_llm import LMStudioLLM  # noqa E402

if llms_to_use == "local_llama":
    task_llm = LMStudioLLM("meta-llama-3.1-8b-instruct", default_temperature, default_max_tokens)
    comparison_llm = LMStudioLLM("meta-llama-3.1-8b-instruct", default_temperature, default_max_tokens)
elif llms_to_use == "kit_llama":
    task_llm = KitLLMReq(os.getenv("LLM_URL"), default_temperature, default_max_tokens)
    comparison_llm = KitLLMReq(os.getenv("LLM_URL"), default_temperature, default_max_tokens)
elif llms_to_use == "local_qwen8":
    task_llm = LMStudioLLM("qwen3-8b", default_temperature, default_max_tokens, no_think=True)
    comparison_llm = LMStudioLLM("qwen3-8b", default_temperature, default_max_tokens, no_think=True)
elif llms_to_use == "local_qwen14":
    task_llm = LMStudioLLM("qwen3-14b", default_temperature, default_max_tokens, no_think=True)
    comparison_llm = LMStudioLLM("qwen3-14b", default_temperature, default_max_tokens, no_think=True)
else:
    raise ValueError(f"Unknown llm_to_use: {llms_to_use}")

if asrs_to_use == "local_whisper_medium":
    asr = LocalWhisperASR("openai/whisper-medium")
elif asrs_to_use == "lecture_translator":
    asr = CloudLectureTranslatorASR()
else:
    raise ValueError(f"Unknown asr_to_use: {asrs_to_use}")


now = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S %z")
log_file_path = os.path.join(base_path, f"data/logs/{now} evaluation_log.json")
print(f"Log file path will be '{log_file_path}'")

eval_pipeline = EvaluationPipeline(
    asr=LocalWhisperASR("openai/whisper-medium"),
    task_llm=task_llm,
    fuzzy_matching_llm=comparison_llm,
    llm_judge=comparison_llm,
    audio_recording_dir_path=audio_file_path,
    verbose_task_execution=False,
    print_progress=True,
    log_file_path=log_file_path,
)

print(f"Startup took {time.time() - script_start_time:.2f} seconds.\n")

# =============================  STARTUP FINISHED  =================================================================


tests = load_test_data(base_path + "tests/data/tests.json")

if random_state is not None:
    interaction_tests_shuffled = pd.Series(tests.interaction).sample(frac=1.0, random_state=random_state).tolist()
    if subset_tests is not None:
        interaction_tests_sample = interaction_tests_shuffled[subset_tests]
    else:
        interaction_tests_sample = interaction_tests_shuffled[:]
else:
    interaction_tests_sample = tests.interaction[:]

# here the tests are run:
I_RES = eval_pipeline.evaluate_individual_tests(interaction_tests_sample)

print("\nAll tests executed.\n")

print(f"Total test duration: {sum(it.time_taken_s for it in I_RES):.2f} seconds.\n")

tmp = pd.DataFrame()
raw = ["crashed" if r.crashed else ("passed" if r.passed else "failed") for r in I_RES]
tmp["abs"] = (pd.Series(raw + ["crashed", "passed", "failed"]).value_counts() - 1).sort_index()
tmp["rel"] = (tmp["abs"] / sum(tmp["abs"]) * 100).round(2)

print(tmp)
print("\nFinished!")
