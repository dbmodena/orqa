from pathlib import Path
import os
import subprocess


class Solr:
    def __init__(self, domain: str):
        if "SOLR_PATH" not in os.environ:
            raise OSError("env variable SOLR_PATH not found.")

        self._solr_path = Path(os.environ["SOLR_PATH"])
        if not self._solr_path.exists():
            raise FileNotFoundError(f"Path {self._solr_path} not exist.")

        self.domain = domain
        self._collection_conf_path = self._solr_path.joinpath(
            "server", "solr", self.domain, "conf"
        )

    def _exec(self, *args):
        subprocess.run(list(args))

    def start(self):
        self._exec("solr", "start", "--user-managed")

    def stop(self):
        self._exec("solr", "stop")

    def _create_collection(self):
        self._exec("solr", "create", "-c", self.domain)

    def create_collection_with_metadata(
        self,
        new_managed_schema_xml: Path,
        new_solrconfig_xml: Path,
        metadata_path: Path,
    ):
        self.start()
        self._create_collection()
        self.stop()

        managed_schema_path = self._collection_conf_path.joinpath("managed-schema.xml")
        solrconfig_path = self._collection_conf_path.joinpath("solrconfig.xml")

        # remove default configuration files
        os.remove(managed_schema_path)
        os.remove(solrconfig_path)

        # copy the custom configuration files
        self._exec("cp", new_managed_schema_xml, managed_schema_path)
        self._exec("cp", new_solrconfig_xml, solrconfig_path)

        self.start()
        self.post_metadata(metadata_path)
        self.stop()

    def post_metadata(self, metadata_path: Path):
        self._exec("solr", "post", "-c", self.domain, str(metadata_path.resolve()))
