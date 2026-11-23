from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block, InteractionBlock
from .layers.time_decoder import TimeDecoder
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class LinearAgentEncoder(nn.Module):
    """
    单层线性智能体历史信息编码器 (LTSF-Linear baseline)
    
    用于验证复杂序列模型（如Mamba）的必要性。
    将历史序列展平后通过单层线性变换映射到嵌入空间，无bias。
    
    特点：
    - 无时序建模能力（无法捕捉时间依赖）
    - 参数量: hist_steps * feat_dim * embed_dim = 50 * 4 * 128 = 25,600
    - 计算复杂度: O(L*D*C)
    """
    def __init__(self, hist_steps=50, feat_dim=4, embed_dim=128):
        super().__init__()
        self.hist_steps = hist_steps
        self.feat_dim = feat_dim
        self.input_dim = hist_steps * feat_dim

        # 单层线性变换，无bias以保持LTSF-Linear的纯粹性
        self.encoder = nn.Linear(self.input_dim, embed_dim, bias=False)

    def forward(self, hist_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hist_feat: [N_valid, L, D] 有效agent的历史特征
                       L=hist_steps, D=feat_dim
        Returns:
            encoded_feat: [N_valid, C] 编码后特征, C=embed_dim
        """
        # 展平时序和特征维度: [N_valid, L, D] -> [N_valid, L*D]
        flattened_feat = hist_feat.flatten(start_dim=1)
        
        # 线性变换: [N_valid, L*D] -> [N_valid, C]
        encoded_feat = self.encoder(flattened_feat)
        
        return encoded_feat

# only 'DeMo'
class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
        hist_steps: int = 50,          # <-- 新增：接收历史步数
        agent_encoder_type: str = 'mamba',  # <-- 新增：编码器切换开关
        use_low_rank: bool = True,  # 添加低秩开关
        low_rank_dim: int = 16,     # 添加低秩维度
        offset_mode: str = None,
        refine_mhca: bool = True,
        refine_mhsa: bool = True,
        refine_mamba: bool = True,
        refine_mamba_bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.agent_encoder_type = agent_encoder_type
        self.embed_dim = embed_dim

        # Agent History Encoder: Mamba (序列模型) vs Linear (LTSF baseline)
        if self.agent_encoder_type == 'mamba':
            # Mamba编码器：具有时序建模能力的状态空间模型
            self.hist_embed_mlp = nn.Sequential(
                nn.Linear(4, 64),
                nn.GELU(),
                nn.Linear(64, embed_dim),
            )
            self.hist_embed_mamba = nn.ModuleList(
                [
                    create_block(
                        d_model=embed_dim,
                        layer_idx=i,
                        drop_path=drop_path,
                        bimamba=False,
                        rms_norm=True,
                    )
                    for i in range(4)
                ]
            )
            self.norm_f = RMSNorm(embed_dim, eps=1e-5)
            self.drop_path = DropPath(drop_path)
        
        elif self.agent_encoder_type == 'linear':
            # Linear编码器：LTSF-Linear baseline，无时序建模能力
            self.linear_encoder = LinearAgentEncoder(
                hist_steps=hist_steps,
                feat_dim=4,  # (x_pos_diff, y_pos_diff, velocity_diff, valid_mask)
                embed_dim=embed_dim
            )
        else:
            raise ValueError(
                f"Unknown agent_encoder_type: '{self.agent_encoder_type}'. "
                f"Supported options: ['mamba', 'linear']"
            )

        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene Context Transformer
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=0.2,
            )
            for i in range(5)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))

        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )

        self.time_embedding_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, embed_dim)
        )

        # self.time_decoder = TimeDecoder()
        # 修改TimeDecoder初始化，传入低秩参数
        self.time_decoder = TimeDecoder(
            future_len=future_steps,
            dim=embed_dim,
            use_low_rank=use_low_rank,
            rank=low_rank_dim,
            offset_mode=offset_mode,
            refine_mhca=refine_mhca,
            refine_mhsa=refine_mhsa,
            refine_mamba=refine_mamba,
            refine_mamba_bidirectional=refine_mamba_bidirectional,
        )

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        return self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, data):
        ###### Scene context encoding ###### 
        # agent encoding
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]
        hist_feat = torch.cat(
            [
                data["x_positions_diff"],
                data["x_velocity_diff"][..., None],
                hist_valid_mask[..., None],
            ],
            dim=-1,
        )

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        # Agent history encoding: 只对有效actor进行编码以节省计算
        if self.agent_encoder_type == 'mamba':
            # Mamba编码器流程
            actor_feat_valid = self.hist_embed_mlp(
                hist_feat[hist_feat_key_valid].contiguous()
            )
            residual = None
            for blk_mamba in self.hist_embed_mamba:
                actor_feat_valid, residual = blk_mamba(actor_feat_valid, residual)
            
            # RMS normalization with residual fusion
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            actor_feat_valid = fused_add_norm_fn(
                self.drop_path(actor_feat_valid),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=True
            )
            # 提取最后一个时间步作为agent表示
            actor_feat_valid = actor_feat_valid[:, -1]

        elif self.agent_encoder_type == 'linear':
            # Linear编码器流程：只对有效actor进行编码
            actor_feat_valid = self.linear_encoder(
                hist_feat[hist_feat_key_valid].contiguous()
            )
        
        # 将有效actor特征填充到完整tensor [B*N, C]
        actor_feat = torch.zeros(
            B * N, actor_feat_valid.shape[-1], 
            device=actor_feat_valid.device, 
            dtype=actor_feat_valid.dtype
        )
        actor_feat[hist_feat_key_valid] = actor_feat_valid
        
        # Reshape为 [B, N, C]
        actor_feat = actor_feat.view(B, N, -1)

        # map encoding
        lane_valid_mask = data["lane_valid_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        # type embedding and position embedding
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed[data["lane_attr"][..., 0].long()]
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        # scene context features
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat(
            [data["x_key_valid_mask"], data["lane_key_valid_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed

        if isinstance(self, StreamModelForecast):
            if "memory_dict" in data and data["memory_dict"] is not None:
                rel_pos = data["origin"] - data["memory_dict"]["origin"]
                rel_ang = (data["theta"] - data["memory_dict"]["theta"] + torch.pi) % (2 * torch.pi) - \
                torch.pi
                rel_ts = data["timestamp"] - data["memory_dict"]["timestamp"]
                memory_pose = torch.cat([
                    rel_ts.unsqueeze(-1), rel_ang.unsqueeze(-1), rel_pos
                ], dim=-1).float().to(x_encoder.device)
                memory_x_encoder = data["memory_dict"]["x_encoder"]
                memory_valid_mask = data["memory_dict"]["x_mask"]
            else:
                memory_pose = x_encoder.new_zeros(x_encoder.size(0), self.pose_dim)
                memory_x_encoder = x_encoder
                memory_valid_mask = key_valid_mask
            cur_pose = torch.zeros_like(memory_pose)

        if isinstance(self, StreamModelForecast) and self.use_stream_encoder:
            new_x_encoder = x_encoder
            for inter in self.interaction:
                new_x_encoder = inter(new_x_encoder, memory_x_encoder, cur_pose, 
                                      memory_pose, key_padding_mask=~memory_valid_mask)
            x_encoder = new_x_encoder * key_valid_mask.unsqueeze(-1) + \
            x_encoder * ~key_valid_mask.unsqueeze(-1)

        #  intra-interaction learning for scene context features
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.norm(x_encoder)

        ###### Trajectory decoding with decoupled queries ###### 
        new_y_hat = None
        new_pi = None
        dense_predict = None
        mode = None

        # outputs of other agents
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)

        # state query initialization
        time = torch.arange(60).long().to(x_encoder.device)
        time = time * 0.1 + 0.1
        time = time.unsqueeze(-1)
        mode = self.time_embedding_mlp(time)
        mode = mode.repeat(x_encoder.size(0), 1, 1)

        # decoder module with decoupled queries
        dense_predict, y_hat, pi, x_mode, new_y_hat, new_pi, mode_dense, scal, scal_new = \
        self.time_decoder(mode, x_encoder, mask=~key_valid_mask)

        if isinstance(self, StreamModelForecast) and self.use_stream_decoder:
            cos, sin = data["theta"].cos(), data["theta"].sin()
            rot_mat = data["theta"].new_zeros(B, 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos

            if "memory_dict" in data and data["memory_dict"] is not None:
                memory_y_hat = data["memory_dict"]["glo_y_hat"]
                memory_x_mode = data["memory_dict"]["x_mode"]
                ori_idx = ((data["timestamp"] - data["memory_dict"]["timestamp"]) / 0.1).long() - 1
                memory_traj_ori = torch.gather(memory_y_hat, 2, ori_idx.reshape(
                    B, 1, -1, 1).repeat(1, memory_y_hat.size(1), 1, memory_y_hat.size(-1)))
                memory_y_hat = torch.bmm(
                    (memory_y_hat - memory_traj_ori).reshape(B, -1, 2).double(), rot_mat
                ).reshape(B, memory_y_hat.size(1), -1, 2).to(torch.float32)
                
                traj_embed = self.traj_embed(y_hat.detach().reshape(B, y_hat.size(1), -1))
                memory_traj_embed = self.traj_embed(memory_y_hat.reshape(B, memory_y_hat.size(1), -1))
                
                for modfus in self.mode_fusion:
                    x_mode = modfus(x_mode, memory_x_mode, cur_pose, memory_pose,
                                    cur_pos_embed=traj_embed,
                                    memory_pos_embed=memory_traj_embed)
                y_hat_diff = self.stream_loc(x_mode).reshape(B, y_hat.size(1), -1, 2)
                y_hat = y_hat + y_hat_diff

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from mode query
            "pi": pi,  # probability output from mode query
            "scal": scal,  # output for Laplace loss from mode query

            "dense_predict": dense_predict,  # trajectory output from state query

            "y_hat_others": y_hat_others,  # trajectory of other agents

            "new_y_hat": new_y_hat,  # final trajectory output
            "new_pi": new_pi,  # final probability output     
            "scal_new": scal_new,  # final output for Laplace loss
        }

        if isinstance(self, StreamModelForecast):
            glo_y_hat = torch.bmm(
                y_hat.detach().reshape(B, -1, 2).double(), torch.inverse(rot_mat)
            ).to(torch.float32)
            glo_y_hat = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)

            memory_dict = {
                "x_encoder": x_encoder,
                "x_mode": x_mode,
                "glo_y_hat": glo_y_hat,
                "x_mask": key_valid_mask,
                "origin": data["origin"],
                "theta": data["theta"],
                "timestamp": data["timestamp"],
            }
            ret_dict["memory_dict"] = memory_dict

        return ret_dict


# integrate 'DeMo' with 'RealMotion'
class StreamModelForecast(ModelForecast):
    def __init__(self, 
                 use_stream_encoder=True,
                 use_stream_decoder=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.use_stream_encoder = use_stream_encoder
        self.use_stream_decoder = use_stream_decoder
        self.embed_dim = kwargs["embed_dim"]
        self.pose_dim = 4
        if self.use_stream_encoder:
            self.interaction = nn.ModuleList(
                InteractionBlock(
                    dim=kwargs["embed_dim"],
                    pose_dim=self.pose_dim,
                    num_heads=kwargs["num_heads"],
                    mlp_ratio=kwargs["mlp_ratio"],
                    qkv_bias=kwargs["qkv_bias"],
                    drop_path=0.2,
                )
                for i in range(1)
            )
        if self.use_stream_decoder:
            self.mode_fusion = nn.ModuleList(
                InteractionBlock(
                    dim=kwargs["embed_dim"],
                    pose_dim=self.pose_dim,
                    num_heads=kwargs["num_heads"],
                    mlp_ratio=kwargs["mlp_ratio"],
                    qkv_bias=kwargs["qkv_bias"],
                    drop_path=0.2,
                )
                for i in range(1)
            )
            self.stream_loc = nn.Sequential(
                nn.Linear(kwargs["embed_dim"], 256),
                nn.GELU(),
                nn.Linear(256, kwargs["embed_dim"]),
                nn.GELU(),
                nn.Linear(kwargs["embed_dim"], kwargs["future_steps"] * 2),
            )
            self.traj_embed = nn.Sequential(
                nn.Linear(kwargs["future_steps"] * 2, kwargs["embed_dim"]),
                nn.GELU(),
                nn.Linear(kwargs["embed_dim"], kwargs["embed_dim"]),
            )

