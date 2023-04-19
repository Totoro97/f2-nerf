//
// Created by ppwang on 2023/3/27.
//

#ifndef SANR_SCATTER_H
#define SANR_SCATTER_H

#include <torch/torch.h>
#include "../../Common.h"

class Scatter {
};

namespace CustomOps {

torch::Tensor ScatterAdd(torch::Tensor emb, torch::Tensor idx, torch::Tensor to_add);
torch::Tensor ScatterIdx(int n_all_pts, torch::Tensor idx_start_end, torch::Tensor emb_idx);
}

#endif //SANR_SCATTER_H
