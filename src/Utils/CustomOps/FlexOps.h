//
// Created by ppwang on 2023/2/11.
//

#ifndef SANR_FLEXOPS_H
#define SANR_FLEXOPS_H

#endif //SANR_FLEXOPS_H

#include <torch/torch.h>
#include "../../Common.h"

namespace FlexOps {

torch::Tensor Sum(torch::Tensor val, torch::Tensor idx_start_end);
torch::Tensor AccumulateSum(torch::Tensor val, torch::Tensor idx_start_end, bool include_this);

}
