import torch
from tools import *


def calc_lower_bound(z_1, z_2, pos, temperature=0.2):
    EOS = 1e-4
    z_1 = z_1.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    z_2 = z_2.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    pos = pos.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    matrix_1 = torch.exp(sim_con(z_1, z_2, temperature))
    matrix_2 = matrix_1.t()
    matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EOS)
    pos = pos.coalesce()

    indices = pos.indices()
    values = pos.values()
    matrix_1_vals_at_pos = matrix_1[indices[0], indices[1]]
    matrix_2_vals_at_pos = matrix_2[indices[0], indices[1]]

    multiplied_values_1 = matrix_1_vals_at_pos * values
    multiplied_values_2 = matrix_2_vals_at_pos * values

    num_rows = pos.shape[0]
    sum_vector_1 = torch.zeros(num_rows, device=pos.device)
    sum_vector_2 = torch.zeros(num_rows, device=pos.device)

    row_indices = indices[0]
    sum_vector_1.index_add_(0, row_indices, multiplied_values_1)
    sum_vector_2.index_add_(0, row_indices, multiplied_values_2)

    
    lori_1 = -torch.log(sum_vector_1 + 1e-6).mean()
    lori_2 = -torch.log(sum_vector_2 + 1e-6).mean()

    return (lori_1 + lori_2) / 2
