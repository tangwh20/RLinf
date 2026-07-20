# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datasets import Features, Sequence

from rlinf.data.lerobot_paths import ensure_hf_datasets_list_feature_compat


def test_hf_datasets_list_feature_compat_is_idempotent():
    list_feature = {
        "feature": {"dtype": "float32", "_type": "Value"},
        "length": 8,
        "_type": "List",
    }

    ensure_hf_datasets_list_feature_compat()
    ensure_hf_datasets_list_feature_compat()

    features = Features.from_dict({"state": list_feature})
    assert isinstance(features["state"], Sequence)
    assert features["state"].length == 8
