from typing import List, Dict


def select_best_backbone(candidates: List[Dict], client_scores: List[List[float]]) -> Dict:
    """
    Select candidate with maximum aggregated energy-aware score.
    client_scores[i][j] = score from client i for candidate j
    """
    num_candidates = len(candidates)
    aggregated = []

    for j in range(num_candidates):
        score_j = sum(client_scores[i][j] for i in range(len(client_scores)))
        aggregated.append(score_j)

    best_idx = max(range(num_candidates), key=lambda j: aggregated[j])
    return candidates[best_idx]
