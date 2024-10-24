from codebleu import calc_codebleu


def get_diversities(code_a, code_b):
    result = calc_codebleu([code_a], [code_b], "python")
    structural_diversity = 1 - result["syntax_match_score"]
    semantic_diversity = 1 - result["dataflow_match_score"]
    return structural_diversity, semantic_diversity


def pairwise_diversity(code_list):
    """Computes pairwise semantic and structural diversity"""
    if len(code_list) < 1:
        return 0, 0

    total_structural = 0
    total_semantic = 0

    # Pairwise diversity.
    for i, code_i in enumerate(code_list):
        for j, code_j in enumerate(code_list):
            if i == j:
                continue
            struct_div, sem_div = get_diversities(code_i, code_j)
            total_structural += struct_div
            total_semantic += sem_div

    count = len(code_list) * len(code_list)
    avg_structural = total_structural / count
    avg_semantic = total_semantic / count

    return avg_structural, avg_semantic


if __name__ == "__main__":
    a = "def add (a , b, c) :\n return a + b + c"
    b = "def sum (first , second) :\n return second + first"
    c = "def sum (x) :\n return x"
    diversity = get_diversities(a, b)
    print(diversity)
    diversity = get_diversities(a, c)
    print(diversity)
    diversity = get_diversities(b, c)
    print(diversity)

    diversity = pairwise_diversity([a, b, c])
    print(diversity)
