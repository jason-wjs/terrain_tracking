from __future__ import annotations

from terrain_tracking.runtime.pair_dataset import PairDatasetRecord


def select_pair_records(
  records: tuple[PairDatasetRecord, ...],
  *,
  pair_filter: tuple[str, ...] = (),
  max_pairs: int | None = None,
) -> tuple[PairDatasetRecord, ...]:
  selected = records
  for filter_expr in pair_filter:
    key, sep, value = filter_expr.partition("=")
    if sep != "=" or not key or not value:
      raise ValueError(f"pair_filter must use key=value syntax, got {filter_expr!r}")
    selected = tuple(record for record in selected if _record_value(record, key) == value)
  if max_pairs is not None:
    if max_pairs <= 0:
      raise ValueError("max_pairs must be positive")
    selected = selected[:max_pairs]
  if not selected:
    raise ValueError("pair selection produced no records")
  return selected


def _record_value(record: PairDatasetRecord, key: str) -> str | None:
  if key == "category":
    return record.category
  if key == "source":
    return record.source
  if key == "terrain.adapter":
    return record.terrain.adapter
  raise ValueError(f"unsupported pair_filter key: {key!r}")


__all__ = ["select_pair_records"]
