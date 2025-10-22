10/13/2025
## 简化了耦合部分的网络架构simplify
- 原DeMo/src/model/layers/time_decoder.py 备份为 DeMo/src/model/layers/time_decoder(backup).py
- 新增 DeMo/src/model/layers/time_decoder.py，实现了：
    - **简化的混合耦合模块** (关键修改点)
        ├─> ✅ cross_block_dense (3层): 与场景上下文最终交互
        ├─> ❌ self_block_dense (删除): 移除大规模混合自注意力
        ├─> ✅ self_block_different_mode (1层): Mode-only极薄交互，防止坍缩，对模式维度k做自注意力
        └─> ✅ dense_embed_mamba (2层): 时序Mamba质检
- output: 结果保存在：
    -lowrank_r32_simplify_seed42
    -lowrank_r32_simplify_seed43

10/20/2025
## 更新了可视化脚本
- 从github仓库的av2的tutorial中提取了视频可视化脚本argo2/vis_argo2.py，结果保存在 argo2/viz_argo2
- 然后转为静态可视化脚本argo2/vis_argo2_static.py脚本,结果保存在argo2/viz_argo2_static
- 基于静态可视化脚本，创建vis_traj_backup.py
- 进一步改进，创建vis_traj.py，成功实现对指定实例的可视化
    - python vis_traj.py --idx 4 --save-dir vis_traj
    - python vis_traj.py --scene_id <场景ID> --save-dir vis_traj
    * 特别的，使用不同的模型时也需要更改src/model/layer中的time_decoder.py （目前有time_decoder_simplify.py和time_decoder_origin.py这两版）

10/21/2025
## 需要在评估时保存详细的实例的结果，用于筛选和对比实例，进而可视化出有明显提升的实例
- eval_details.py用于评估时保存详细的实例评估结果
- 完成了对lowrank版本的测试，结果位于outputs/lowrank_r32_simplify_results_detail/seed44/20251021-170209/instance_detailed_results.pkl
- 完成了对original版本的测试，结果位于DeMo/outputs/original_results_detail/seed44/20251021-165644/instance_detailed_results.pkl
- 通过compare_instance.py，挑选出ade和fde相差最大的前5个场景，汇总到outputs/viz_instance_compare/scenario_ids.txt
- 通过python vis_traj.py --scene_id <场景ID> --save-dir viz_traj或者viz_traj_origin将这些差异化的实例进行可视化
    - python vis_traj.py --scene_id 6a28ba10-4ba7-4664-b85f-9adb89729225 --save-dir viz_traj --checkpoint outputs/lowrank_r32_simplify_seed/seed_44/20251020-233228/checkpoints/epoch\=59.ckpt