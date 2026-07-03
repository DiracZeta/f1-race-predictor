"""Gold layer: feature tables for modeling.

Aggregate and engineer race-level features (recent form, grid position,
constructor performance, circuit history, ...) into the table the model uses.
This layer is the stable contract between data engineering and modeling.
"""


def build_gold():
    raise NotImplementedError
