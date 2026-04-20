from __future__ import annotations

import types
import unittest

import numpy as np

from hy_models import coreml_a8_stateful


class _FakeModel:
    def __init__(self, *, state_count: int, make_state_result: object | None = object()) -> None:
        self._spec = types.SimpleNamespace(
            description=types.SimpleNamespace(state=[object() for _ in range(state_count)])
        )
        self._make_state_result = make_state_result
        self.predict_calls: list[dict[str, object]] = []
        self.make_state_calls = 0

    def get_spec(self):
        return self._spec

    def make_state(self):
        self.make_state_calls += 1
        return self._make_state_result

    def predict(self, inputs, state=None):
        self.predict_calls.append({"inputs": inputs, "state": state})
        return {"logits": np.asarray([1.0], dtype=np.float32)}


class PredictWithOptionalStateTests(unittest.TestCase):
    def test_stateful_model_uses_predict_with_state(self) -> None:
        fake_model = _FakeModel(state_count=1, make_state_result="MLSTATE")
        payload = {"input_ids": np.asarray([[1, 2, 3]], dtype=np.int32)}
        _ = coreml_a8_stateful._predict_with_optional_state(fake_model, payload)

        self.assertEqual(fake_model.make_state_calls, 1)
        self.assertEqual(len(fake_model.predict_calls), 1)
        self.assertEqual(fake_model.predict_calls[0]["inputs"], payload)
        self.assertEqual(fake_model.predict_calls[0]["state"], "MLSTATE")

    def test_stateless_model_uses_predict_without_state(self) -> None:
        fake_model = _FakeModel(state_count=0)
        payload = {"input_ids": np.asarray([[4, 5]], dtype=np.int32)}
        _ = coreml_a8_stateful._predict_with_optional_state(fake_model, payload)

        self.assertEqual(fake_model.make_state_calls, 0)
        self.assertEqual(len(fake_model.predict_calls), 1)
        self.assertEqual(fake_model.predict_calls[0]["inputs"], payload)
        self.assertIsNone(fake_model.predict_calls[0]["state"])

    def test_stateful_model_raises_when_make_state_returns_none(self) -> None:
        fake_model = _FakeModel(state_count=2, make_state_result=None)
        payload = {"input_ids": np.asarray([[7]], dtype=np.int32)}

        with self.assertRaisesRegex(RuntimeError, "got None"):
            coreml_a8_stateful._predict_with_optional_state(fake_model, payload)


if __name__ == "__main__":
    unittest.main()
