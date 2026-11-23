# 原始版本，仅在demo的基础上添加了可使用低秩分解适配器模块的选项，但这个版本只用于进行原始DeMo模型的对比实验，实际上不用低秩分解
import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F
from .mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

class LowRankAdapter(nn.Module):
    """低秩分解适配器，为每个模式生成轻量的状态查询增量"""
    def __init__(self, dim=128, rank=16, num_modes=6):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_modes = num_modes
        
        # 低秩分解: ΔS_k = A_k * B^T
        # A_k: [num_modes, T_s, rank] - 每个模式独有
        # B: [dim, rank] - 所有模式共享
        self.mode_specific_A = nn.Parameter(torch.randn(num_modes, 60, rank) * 0.01)
        self.shared_B = nn.Parameter(torch.randn(dim, rank) * 0.01)
        
        # 可选：为增量添加缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, global_states, mode_idx=None):
        """
        Args:
            global_states: [B, T_s, C] - 全局状态查询
            mode_idx: [B] or None - 如果为None，返回所有模式的增量
        
        Returns:
            如果mode_idx指定: [B, T_s, C] - 特定模式的状态查询
            否则: [B, K, T_s, C] - 所有模式的状态查询
        """
        B, T_s, C = global_states.shape
        
        if mode_idx is not None:
            # 获取特定模式的A矩阵
            A_k = self.mode_specific_A[mode_idx]  # [B, T_s, rank]
            delta_S_k = torch.matmul(A_k, self.shared_B.T)  # [B, T_s, C]
            return global_states + self.scale * delta_S_k
        else:
            # 计算所有模式的增量
            # A: [K, T_s, rank], B: [C, rank]
            # ΔS: [K, T_s, C]
            all_deltas = torch.matmul(self.mode_specific_A, self.shared_B.T)  # [K, T_s, C]
            
            # 扩展global_states并添加增量
            global_states_expand = global_states.unsqueeze(1).expand(-1, self.num_modes, -1, -1)  # [B, K, T_s, C]
            all_deltas_expand = all_deltas.unsqueeze(0).expand(B, -1, -1, -1)  # [B, K, T_s, C]
            
            return global_states_expand + self.scale * all_deltas_expand

class GMMPredictor_dense(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor_dense, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2)
        )
    
    def forward(self, input):
        res = self.gaussian(input)
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        input = input.max(dim=2)[0]  
        score = self.score(input).squeeze(-1)

        return res, score, scal


class GMMPredictor(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        self.scale = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
    
    def forward(self, input):
        B, M, _ = input.shape
        res = self.gaussian(input).view(B, M, self._future_len, 2) 
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        scal = scal.view(B, M, self._future_len, 2) 
        score = self.score(input).squeeze(-1)

        return res, score, scal
    

class TimeDecoder(nn.Module):
    def __init__(self, future_len=60, dim=128, use_low_rank=True, rank=16):
        super(TimeDecoder, self).__init__()
        
        self.use_low_rank = use_low_rank
        self.num_modes = 6  # 模式数量

        ###### State Consistency Module ######
        # state cross attention
        self.cross_block_time = nn.ModuleList(
            Cross_Block()
            for i in range(2)
        )

        # state bidirectional mamba
        self.timequery_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )  
        self.timequery_norm_f = RMSNorm(dim, eps=1e-5)
        self.timequery_drop_path = DropPath(0.2)

        # MLP for state query
        self.dense_predict = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        ###### 低秩分解模块 ######
        if self.use_low_rank:
            self.low_rank_adapter = LowRankAdapter(dim=dim, rank=rank, num_modes=self.num_modes)

        ###### Mode Localization Module ######
        # mode self attention
        self.self_block_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # mode cross attention
        self.cross_block_mode = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode query initialization
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        # MLP for mode query
        self.predictor = GMMPredictor(future_len)

        ###### Hybrid Coupling Module ######
        # hybrid self attention
        self.self_block_dense = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # hybrid cross attention
        self.cross_block_dense = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode self attention for hybrid spatiotemporal queries
        self.self_block_different_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # state bidirectional mamba for hybrid spatiotemporal queries 
        self.dense_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=True,  
                    rms_norm=True,  
                )
                for i in range(2)
            ]
        )
        self.dense_norm_f = RMSNorm(dim, eps=1e-5)
        self.dense_drop_path = DropPath(0.2)

        # MLP for final output
        self.predictor_dense = GMMPredictor_dense(future_len)

    def forward(self, mode, encoding, mask=None):
        # Dynamic state consistency - 学习全局状态查询
        for blk in self.cross_block_time:
            mode = blk(mode, encoding, key_padding_mask=mask)
        
        residual = None
        for blk_mamba in self.timequery_embed_mamba:
            mode, residual = blk_mamba(mode, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.timequery_norm_f, RMSNorm) else layer_norm_fn
        mode = fused_add_norm_fn(
            self.timequery_drop_path(mode),
            self.timequery_norm_f.weight,
            self.timequery_norm_f.bias,
            eps=self.timequery_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )
        
        # 全局状态查询的预测（可选）
        dense_pred = self.dense_predict(mode)
        
        # 保存全局状态查询
        global_state_query = mode  # [B, T_s, C]
        
        # 使用低秩分解为每个模式生成特定的状态查询
        if self.use_low_rank:
            # 生成所有模式的状态查询: S_k = S + ΔS_k
            mode_specific_states = self.low_rank_adapter(global_state_query)  # [B, K, T_s, C]
        else:
            # 原始方案：所有模式共享同一状态查询
            mode_specific_states = global_state_query.unsqueeze(1).expand(-1, self.num_modes, -1, -1)
        
        # Directional intention localization
        multi_modal_query = self.multi_modal_query_embedding(self.modal)
        mode_query = encoding[:, 0]
        mode = mode_query[:, None] + multi_modal_query

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.self_block_mode:
            mode = blk(mode)

        y_hat, pi, scal = self.predictor(mode)

        # Hybrid query coupling - 使用模式特定的状态查询
        B, K, T, C = mode_specific_states.shape
        
        # mode: [B, K, C], mode_specific_states: [B, K, T, C]
        # 组合方式：每个模式使用自己的状态查询
        mode_dense = mode[:, :, None, :] + mode_specific_states  # [B, K, T, C]
        
        # 后续处理保持不变
        mode_dense = mode_dense.reshape(B, -1, C)
        for blk in self.cross_block_dense:
            mode_dense = blk(mode_dense, encoding, key_padding_mask=mask)
        for blk in self.self_block_dense:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, K, T, C)
        
        mode_dense = mode_dense.transpose(1, 2).reshape(-1, K, C)
        for blk in self.self_block_different_mode:
            mode_dense = blk(mode_dense)
        mode_dense = mode_dense.reshape(B, -1, K, C).transpose(1, 2)

        mode_dense = mode_dense.reshape(-1, T, C)
        residual = None
        for blk_mamba in self.dense_embed_mamba:
            mode_dense, residual = blk_mamba(mode_dense, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.dense_norm_f, RMSNorm) else layer_norm_fn
        mode_dense = fused_add_norm_fn(
            self.dense_drop_path(mode_dense),
            self.dense_norm_f.weight,
            self.dense_norm_f.bias,
            eps=self.dense_norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )
        mode_dense = mode_dense.reshape(B, K, T, C)

        y_hat_new, pi_new, scal_new = self.predictor_dense(mode_dense)

        return dense_pred, y_hat, pi, mode, y_hat_new, pi_new, mode_dense, scal, scal_new
