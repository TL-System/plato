from math import comb as nCr


def expected_amount(k, n):
    """
    k number of bins, n batch size
    """
    total_num = nCr(k + n - 1, k - 1)  # Total number of configs
    weight = 0
    for i in range(1, n - 1):
        temp = i * nCr(k, i)
        temp2 = 0
        for j in range(1, (n - i) // 2 + 1):
            temp2 += nCr(k - i, j) * nCr(n - i - j - 1, j - 1)
        weight += temp * temp2
    adjustment1 = n * nCr(k, n)  # First term in r(n,k)
    weight += adjustment1
    return weight / total_num - n / k  # Second adjustment term in r(n,k)


def one_shot_guarantee(k, n):
    """
    k number of bins, n batch size
    """
    total_num = nCr(k + n - 1, k - 1)  # Total number of configs
    weight = 0
    weight += nCr(n + k - 3, k - 2)
    return weight / total_num


if __name__ == "__main__":
    print(expected_amount(128, 5))
