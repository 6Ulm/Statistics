import json
import pathlib
import unittest


DATA = pathlib.Path(__file__).parents[1] / "public" / "data"


class ScientificDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "manifest.json").read_text())

    def test_manifest_has_retained_rho_values_and_shared_labels(self):
        self.assertEqual(self.manifest["format"], "coclust-browser-v1")
        self.assertEqual(len(self.manifest["rhos"]), 17)
        self.assertTrue(all(meta["value"] < 0.1 for meta in self.manifest["rhos"]))
        self.assertTrue(all(meta["mass"] > 0 for meta in self.manifest["rhos"]))
        self.assertEqual(self.manifest["shape"], [2459, 1881])
        self.assertEqual(len(self.manifest["rowNames"]), 2459)
        self.assertEqual(len(self.manifest["columnNames"]), 1881)
        self.assertEqual(len(set(self.manifest["rowNames"])), 2459)
        self.assertEqual(len(set(self.manifest["columnNames"])), 1881)

    def test_every_retained_rho_asset_is_consistent_and_discards_original_labels(self):
        for meta in self.manifest["rhos"]:
            payload = json.loads((DATA / meta["file"]).read_text())
            matrix = payload["matrix"]
            self.assertEqual(len(matrix["indptr"]), 2460)
            self.assertEqual(matrix["indptr"][-1], meta["nnz"])
            self.assertEqual(len(matrix["indices"]), meta["nnz"])
            self.assertEqual(len(matrix["data"]), meta["nnz"])
            self.assertEqual(sorted(map(int, payload["clusters"])), meta["clusterIds"])
            self.assertEqual(len(payload["biclusters"]), len(meta["clusterIds"]))
            self.assertEqual(
                [block["cluster"] for block in payload["biclusters"]],
                list(range(len(payload["biclusters"]))),
            )
            for block in payload["biclusters"]:
                self.assertNotIn("rowCluster", block)
                self.assertNotIn("columnCluster", block)
                self.assertEqual(len(block["mouseGenes"]), block["nMouse"])
                self.assertEqual(len(block["humanGenes"]), block["nHuman"])


if __name__ == "__main__":
    unittest.main()
