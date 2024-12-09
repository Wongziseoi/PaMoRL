import math
import torch


def binary_return_fn(cur_i, cur_j):
    coef_i, in_i = cur_i
    coef_j, in_j = cur_j
    return coef_i * coef_j, coef_j * in_i + in_j


def parallel_eligibility_trace(reward, value, next_value, p_cont, lam):
    ones = torch.ones_like(reward)
    p_cont, lam = p_cont * ones, lam * ones
    lam = torch.cat((lam[1:], ones[:1]), dim=0)

    delta = reward + p_cont * next_value - value
    flipped_delta = delta.flip(dims=(0,))
    flipped_lam = (p_cont * lam).flip(dims=(0,))

    residual = odd_even_parallel_scan(
        [flipped_lam, flipped_delta], binary_return_fn)
    returns = value + residual[1].flip(dims=(0,))
    return returns


def parallel_lambda_return(reward, value, next_value, p_cont, lam):
    ones = torch.ones_like(reward)
    p_cont, lam = p_cont * ones, lam * ones

    delta = reward + p_cont * next_value * (1 - lam)
    last = delta[-1:] + p_cont[-1:] * lam[-1:] * next_value[-1:]
    delta = torch.cat((delta[:-1], last), dim=0)

    flipped_delta = delta.flip(dims=(0,))
    flipped_lam = (p_cont * lam).flip(dims=(0,))

    returns = odd_even_parallel_scan(
        [flipped_lam, flipped_delta], binary_return_fn)
    returns = returns[1].flip(dims=(0,))
    return returns


# Kogge-Stone Parallel Scan: O(log2N)
# input: List of [L, B, D] Tensors
def kogge_stone_parallel_scan(inputs, operator):
    Length = inputs[0].shape[0]
    Times = math.ceil(math.log2(Length))

    for i in range(Times):
        interval = int(2 ** i)
        outputs = operator(
            (input[:-interval] for input in inputs),
            (input[interval:] for input in inputs)
        )
        inputs = [
            torch.cat((input[:interval], output), dim=0)
            for (input, output) in zip(inputs, outputs)
        ]
    return inputs


def interleave(odd, even):
    padded_odd = torch.cat((odd, torch.zeros_like(odd[-1:])), dim=0)
    outputs = torch.stack((even, padded_odd[:even.shape[0]]), dim=1)
    outputs = outputs.flatten(0, 1)[:(odd.shape[0] + even.shape[0])]
    return outputs


# Odd/Even Parallel Scan: O(2log2N)
# Recursive implementation
def odd_even_parallel_scan(inputs, operator):
    Length = inputs[0].shape[0]

    if Length < 2:
        return inputs

    reduced_inputs = operator(
        (input[:-1][0::2] for input in inputs), 
        (input[1::2] for input in inputs)
    )
    odd_inputs = odd_even_parallel_scan(reduced_inputs, operator)

    if Length % 2 == 0:
        even_inputs = operator(
            (input[:-1] for input in odd_inputs),
            (input[2::2] for input in inputs)
            )
    else:
        even_inputs = operator(
            (input for input in odd_inputs),
            (input[2::2] for input in inputs)
        )

    even_inputs = [
        torch.cat((input[0:1], even_input), dim=0)
        for (input, even_input) in zip(inputs, even_inputs)
    ]

    outputs = [
        interleave(odd_input, even_input) 
        for (even_input, odd_input) in zip(even_inputs, odd_inputs)
    ]
    return outputs