from __future__ import annotations

import glob
import json
from pathlib import Path


class DorabotLogReader:
    supported_schemas = {"bc_v1", "joint_bc_v1"}

    def read(self, patterns: list[str]) -> list[dict]:
        paths = []
        for pattern in patterns:
            matches = sorted(glob.glob(pattern))
            paths.extend(matches or [pattern])
        rows = []
        for path in paths:
            with Path(path).open() as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("schema_version") not in self.supported_schemas:
                        continue
                    row["_source_file"] = str(path)
                    row["_source_line"] = line_no
                    rows.append(row)
        return rows
