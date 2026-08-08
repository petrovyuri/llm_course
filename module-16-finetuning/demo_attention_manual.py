import torch
from attention_simple import compute_attention_step_by_step, SimpleSelfAttention


def demo_with_manual_vectors():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # The
            [0.55, 0.87, 0.66],  # cat
            [0.57, 0.85, 0.64],  # loved
        ],
        dtype=torch.float32,
    )
    token_names = ["The", "cat", "loved"]
    results = compute_attention_step_by_step(inputs, token_names=token_names)

    print("\n=== ПРОВЕРКА СОВПАДЕНИЯ С РУЧНЫМИ ВЫЧИСЛЕНИЯМИ ===")
    print("scores[0,0] ожид 0.9995 =>", round(results["attention_scores"][0, 0].item(), 4))
    print("scores[1,2] ожид 1.4754 =>", round(results["attention_scores"][1, 2].item(), 4))
    print("сумма весов строки 1 ожид 1.0 =>", round(results["attention_weights"][1].sum().item(), 4))

    print("\n=== ПРОВЕРКА ЧЕРЕЗ КЛАСС SimpleSelfAttention ===")
    module = SimpleSelfAttention()
    ctx, w, sc = module(inputs)
    print("context совпадает:", torch.allclose(results["context_vectors"], ctx, atol=1e-6))
    print("weights совпадает:", torch.allclose(results["attention_weights"], w, atol=1e-6))


def demo_matrix_operations():
    inputs = torch.tensor([[0.43,0.15,0.89],[0.55,0.87,0.66],[0.57,0.85,0.64]], dtype=torch.float32)
    n = inputs.shape[0]
    loops = torch.empty(n, n)
    for i in range(n):
        for j in range(n):
            loops[i, j] = torch.dot(inputs[i], inputs[j])
    mat = inputs @ inputs.T
    print("\n=== Циклы vs матрицы ===")
    print("совпадают:", torch.allclose(loops, mat, atol=1e-6))


if __name__ == "__main__":
    demo_with_manual_vectors()
    demo_matrix_operations()
    print("\nДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
