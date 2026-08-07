import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import load_config


class ConfigEnvironmentTest(unittest.TestCase):
    def test_load_config_expands_environment_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(
                "uri: ${MILVUS_URI}\noutput: ${MNTPNT}/artifact\n",
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"MILVUS_URI": "http://milvus:19530", "MNTPNT": "/data"},
            ):
                config = load_config(str(config_path))
        self.assertEqual(config["uri"], "http://milvus:19530")
        self.assertEqual(config["output"], "/data/artifact")

    def test_load_config_rejects_unset_environment_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("uri: ${MILVUS_URI}\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "unset environment variable"):
                    load_config(str(config_path))

    def test_trace_workload_examples_are_loadable(self):
        repository = Path(__file__).resolve().parents[1]
        environment = {
            "MNTPNT": "/data/ragperf",
            "MILVUS_URI": "http://milvus:19530",
            "RAG_DEVICE": "cuda:0",
            "GENERATION_DEVICE": "cuda:1",
        }
        configs = (
            "milvus_trace_text.yaml",
            "milvus_trace_image.yaml",
            "milvus_audio.yaml",
        )
        with mock.patch.dict(os.environ, environment):
            for name in configs:
                config = load_config(str(repository / "config" / name))
                self.assertEqual(config["sys"]["vector_db"]["type"], "milvus")
                self.assertTrue(config["sys"]["vector_db"]["trace"]["enabled"])
                self.assertTrue(config["sys"]["vector_db"]["trace"]["output_dir"].startswith("/data/ragperf/"))


if __name__ == "__main__":
    unittest.main()
