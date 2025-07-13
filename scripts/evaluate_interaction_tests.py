#
#
#
# =============================  VARIABLES  ========================================================================

# random state for the test shuffling. Use None to disable shuffling (not recommended, may lead to autocorrelation)
random_state: int | None = None  # 2308421

# You can specify a subset of the tests to run if you want to.
# If not None, tests are only included if any of their queries contains any of the filter substrings.
query_filter: set[str] | None = None  # {"* Replace all ment"}

# Filter tests by name. Must be the exact name.
name_filter: set[str] | None = None  # {"add_card"}

# If not None, only the given slice of the tests are used. Applied after shuffling.
subset_indexes: slice | None = None  # slice(0, 4)

# number of times to run each test.
iterations: int = 1

# options: 'local_llama', 'kit_llama', 'local_qwen8', 'local_qwen14'
llms_to_use: str = "local_llama"

# Where are the audio files? If asr should be skipped (only using text prompts), set to None.
audio_file_path: str | None = "./data/recording_data/fabian"

# options: 'local_whisper_medium', 'lecture_translator'
asr_to_use: str = "local_whisper_medium"

# levenshtein distance settings for question/answer matching
# distance: distance between two questions/answers to be considered similar.
# ratio: distance / max(s1, s2). 1 means completely different strings, 0 means exact match.
# If both are set, both thresholds need to be met. If both are not set, only hard matching is used.
max_levenshtein_distance: int | None = 8
max_levenshtein_ratio: float | None = 0.201  # more than 1/5

default_temperature: float = 0.0
default_max_tokens: int = 1000

# set a maximum number of states to visit. Use this to test if the early states work well without having to wait for
# later states.
max_states: int | None = None

# If dry run: Only output the final test sample, do not actually run tests. No log file created.
dry_run: bool = False

# ==================================================================================================================

print(
    f"""The configuration is:
random_state: {random_state}
query_filter: {query_filter}
name_filter: {name_filter}
subset_indexes: {subset_indexes}
iterations: {iterations}
llms_to_use: {llms_to_use}
audio_file_path: {audio_file_path}
asr_to_use: {asr_to_use}
max_levenshtein_distance: {max_levenshtein_distance}
max_levenshtein_ratio: {max_levenshtein_ratio}
default_temperature: {default_temperature}
default_max_tokens: {default_max_tokens}
max_states: {max_states}
dry_run: {dry_run}
"""
)

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

# test that the current working dir is "the-curator"
if os.path.basename(os.path.abspath(".")) != "the-curator":
    raise RuntimeError("This script must be run from the 'the-curator' directory.")

# must happen before project imports!
if not (os.path.exists(".env") and os.path.exists(".env.db")):
    raise FileNotFoundError("No .env or .env.db found.")

load_dotenv(".env")
load_dotenv(".env.db")

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
    comparison_llm = LMStudioLLM("meta-llama-3.1-8b-instruct", 0.0, 10)
elif llms_to_use == "kit_llama":
    task_llm = KitLLMReq(os.getenv("LLM_URL"), default_temperature, default_max_tokens)
    comparison_llm = KitLLMReq(os.getenv("LLM_URL"), 0.0, 10)
elif llms_to_use == "local_qwen8":
    task_llm = LMStudioLLM("qwen3-8b", default_temperature, default_max_tokens, no_think=True)
    comparison_llm = LMStudioLLM("qwen3-8b", 0.0, 20, no_think=True)  # needs more tokens for empty thinking block.
elif llms_to_use == "local_qwen14":
    task_llm = LMStudioLLM("qwen3-14b", default_temperature, default_max_tokens, no_think=True)
    comparison_llm = LMStudioLLM("qwen3-14b", 0.0, 20, no_think=True)
else:
    raise ValueError(f"Unknown llm_to_use: {llms_to_use}")

if asr_to_use == "local_whisper_medium":
    asr = LocalWhisperASR("openai/whisper-medium")
elif asr_to_use == "lecture_translator":
    asr = CloudLectureTranslatorASR()
else:
    raise ValueError(f"Unknown asr_to_use: {asr_to_use}")


now = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S %z")
log_file_path = f"data/logs/{now} evaluation_log.json"
print(f"Log file path will be '{log_file_path}'")

eval_pipeline = EvaluationPipeline(
    asr=asr,
    task_llm=task_llm,
    fuzzy_matching_llm=comparison_llm,
    llm_judge=comparison_llm,
    max_levenshtein_distance=max_levenshtein_distance,
    max_levenshtein_ratio=max_levenshtein_ratio,
    max_states=max_states,
    audio_recording_dir_path=audio_file_path,
    verbose_task_execution=False,
    print_progress=True,
    log_file_path=log_file_path,
)

print(f"Startup took {time.time() - script_start_time:.2f} seconds.\n")

# =============================  STARTUP FINISHED  =================================================================


tests = load_test_data("tests/data/tests.json")

# get interaction tests
interaction_tests = tests.interaction[:]

# filter if wanted
if name_filter is not None:
    interaction_tests = [it for it in interaction_tests if it.name in name_filter]

if query_filter is not None:
    interaction_tests = [it for it in interaction_tests if any(s in q for s in query_filter for q in it.queries)]

# shuffle if wanted
if random_state is not None:
    interaction_tests = pd.Series(interaction_tests).sample(frac=1.0, random_state=random_state).tolist()

# get slice if wanted
if subset_indexes is not None:
    interaction_tests = interaction_tests[subset_indexes]

# run test multiple times if wanted
if iterations > 1:
    interaction_tests = iterations * interaction_tests

print(f"Tests loaded after: {time.time() - script_start_time:.2f} seconds.\n")

# test dry run
if dry_run:
    print("Dry run: Only output the final test list, do not actually run tests.")
    print("--------------------------------------------------------------------")
    print("\n\n".join(str(it) for it in interaction_tests))
    print("--------------------------------------------------------------------")
    exit(0)

# here the tests are run:
I_RES = eval_pipeline.evaluate_individual_tests(interaction_tests)

print("\nAll tests executed.\n")

print(f"Total test duration: {sum(it.time_taken_s for it in I_RES):.2f} seconds.\n")

tmp = pd.DataFrame()
raw = ["crashed" if r.crashed else ("passed" if r.passed else "failed") for r in I_RES]
tmp["abs"] = (pd.Series(raw + ["crashed", "passed", "failed"]).value_counts() - 1).sort_index()
tmp["rel"] = (tmp["abs"] / sum(tmp["abs"]) * 100).round(2)

print(tmp)
print("\nFinished!")
