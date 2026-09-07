# FoveaCast versus heuristic prominence

## Run provenance

Git SHA: `b303f4da942679eeb4f41534ff498613350daba6`; working tree dirty: `True`; corpus SHA-256: `ee5fb059f0179d80d550995215c02a1df96c661a818e85710f9e9058eac3f7b8`; benchmark source SHA-256: `e7c92006e59e004ddd5f84648c538528a8f25667efed5eeafa357e40736c6ce1`.

## Calibration and holdout

The corpus contains 24 controlled cases. Fusion was calibrated only on the calibration split and evaluated on the disjoint holdout split.

| Split | Provider | Top-1 | MRR | NDCG@3 | Pairwise | Mean ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| calibration | heuristic | 0.083 | 0.444 | 0.575 | 0.350 | 0.132 |
| calibration | foveacast | 0.083 | 0.361 | 0.465 | 0.217 | 910.223 |
| calibration | hybrid | 0.083 | 0.444 | 0.575 | 0.350 | 910.355 |
| holdout | heuristic | 0.083 | 0.451 | 0.593 | 0.367 | 0.142 |
| holdout | foveacast | 0.167 | 0.403 | 0.520 | 0.267 | 926.975 |
| holdout | hybrid | 0.083 | 0.451 | 0.593 | 0.367 | 927.118 |

## Recommendation

**keep-both**: Neither pure provider dominates and calibrated fusion does not clear the material-improvement threshold.

This is controlled synthetic evidence with human-authored relevance labels, not eye-tracking or a real-user usability study.
