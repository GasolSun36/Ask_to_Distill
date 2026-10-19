import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


# `livecodebench/code_generation_lite` is a script-based HF dataset, and datasets>=4
# dropped loading scripts (and the `trust_remote_code` kwarg) entirely. Read the jsonl
# shards straight off the hub instead; the shard->version mapping below is copied from
# the upstream `code_generation_lite.py` builder so the problem sets are identical.
_LCB_SHARDS = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
_LCB_VERSION_SHARDS = {f"release_v{i}": _LCB_SHARDS[:i] for i in range(1, 7)}
_LCB_VERSION_SHARDS["release_latest"] = list(_LCB_SHARDS)
for _i in range(1, 7):
    _LCB_VERSION_SHARDS[f"v{_i}"] = [_LCB_SHARDS[_i - 1]]
    for _j in range(_i + 1, 7):
        _LCB_VERSION_SHARDS[f"v{_i}_v{_j}"] = _LCB_SHARDS[_i - 1 : _j]


# a2d: pin the dataset snapshot the paper's numbers were computed on. Without a
# revision, hf_hub_download resolves `main`, so a later upstream push would silently
# change the problem set. LCB_DATASET_REVISION overrides it (e.g. `main`).
_LCB_DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"


def _load_lcb_jsonl(release_version: str) -> list[dict]:
    import os
    from huggingface_hub import hf_hub_download

    if release_version not in _LCB_VERSION_SHARDS:
        raise ValueError(
            f"{release_version} not in {sorted(_LCB_VERSION_SHARDS)}"
        )
    revision = os.environ.get("LCB_DATASET_REVISION", _LCB_DATASET_REVISION)
    rows = []
    for shard in _LCB_VERSION_SHARDS[release_version]:
        path = hf_hub_download(
            "livecodebench/code_generation_lite", shard, repo_type="dataset",
            revision=revision,
        )
        with open(path) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    dataset = [CodeGenerationProblem(**p) for p in _load_lcb_jsonl(release_version)]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
