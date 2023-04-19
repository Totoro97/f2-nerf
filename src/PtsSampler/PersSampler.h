//
// Created by ppwang on 2022/9/26.
//

#ifndef SANR_PERSSAMPLER_H
#define SANR_PERSSAMPLER_H
#include "PtsSampler.h"
#include "Eigen/Eigen"

#define INIT_NODE_STAT 1000
#define N_PROS 12
#define PersMatType Eigen::Matrix<float, 2, 4, Eigen::RowMajor>
#define TransWetType Eigen::Matrix<float, 3, N_PROS, Eigen::RowMajor>

struct alignas(32) TransInfo {
  PersMatType w2xz[N_PROS];
  TransWetType weight;
  Wec3f center;
  float dis_summary;
};

struct alignas(32) TreeNode {
  Wec3f center;
  float side_len;
  int parent;
  int childs[8];
  bool is_leaf_node;
  int trans_idx;
};

struct alignas(32) EdgePool {
  int t_idx_a;
  int t_idx_b;
  Wec3f center;
  Wec3f dir_0;
  Wec3f dir_1;
};

class PersOctree {
  using Tensor = torch::Tensor;
public:
  PersOctree(int max_depth, float bbox_side_len, float split_dist_thres,
             const Tensor& c2w, const Tensor& w2c, const Tensor& intri, const Tensor& bound);

  std::vector<int> CalcVisiCams(const Tensor& pts);
  void ConstructTreeNode(int u, int depth, Wec3f center, float side_len);
  TransInfo ConstructTrans(const Tensor& rand_pts,
                           const Tensor& c2w,
                           const Tensor& intri,
                           const Tensor& center); // Share intri;
  void ProcOctree(bool compact, bool subdivide, bool brute_force);
  void MarkInvisibleNodes();

  void ConstructEdgePool();

  int max_depth_;
  Tensor c2w_, w2c_, intri_, bound_;
  float bbox_side_len_;
  float split_dist_thres_;

  std::vector<TreeNode> tree_nodes_;
  Tensor tree_nodes_gpu_;
  Tensor tree_weight_stats_, tree_alpha_stats_;
  Tensor tree_visit_cnt_;
  Tensor node_search_order_;

  std::vector<TransInfo> pers_trans_;
  Tensor pers_trans_gpu_;

  std::vector<EdgePool> edge_pool_;
  Tensor edge_pool_gpu_;
};

class PersSampler : public PtsSampler {
  using Tensor = torch::Tensor;
public:
  PersSampler(GlobalDataPool* global_data_pool);
  SampleResultFlex GetSamples(const Tensor& rays_o, const Tensor& rays_d, const Tensor& bounds) override;

  void VisOctree();
  void UpdateOctNodes(const SampleResultFlex& sample_result,
                      const Tensor& sampled_weights,
                      const Tensor& sampled_alpha) override;

  std::vector<Tensor> States() override;
  int LoadStates(const std::vector<Tensor>& states, int idx) override;

  std::tuple<Tensor, Tensor> GetEdgeSamples(int n_pts);
  std::unique_ptr<PersOctree> pers_octree_;
  std::vector<int> sub_div_milestones_;
  int compact_freq_;
  int max_oct_intersect_per_ray_;
  float global_near_;
  float sample_l_;
  bool scale_by_dis_;
};

#endif //SANR_PERSSAMPLER_H
