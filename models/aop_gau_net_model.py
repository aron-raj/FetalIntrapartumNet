import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as grad_ckpt
DEBUG: bool = False
_shape_printed: bool = False
def _dbg_shape(tag: str, t: torch.Tensor) -> None:
    global _shape_printed
    if DEBUG and not _shape_printed:
        print(f"  [DBG] {tag:<34s} {tuple(t.shape)}")
def _dbg_mem(tag: str = "") -> None:
    if DEBUG and torch.cuda.is_available():
        mb = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"  [DBG] VRAM {tag}: {mb:.1f} MB")
def mark_shapes_done() -> None:
    global _shape_printed
    _shape_printed = True
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, padding: int = 1, bias: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size,
                            padding=padding, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_dw: bool = False,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        conv = (DepthwiseSeparableConv(in_ch, out_ch, kernel_size, padding)
                if use_dw else
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False))
        self.block = nn.Sequential(conv, nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_dw: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch,  out_ch, use_dw),
            ConvBNReLU(out_ch, out_ch, use_dw),
        )
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip
class PositionalEncoding2D(nn.Module):
    def __init__(self, coord_dim: int = 2, feat_dim: int = 128):
        super().__init__()
        max_cd         = max(1, int(math.floor(0.25 * feat_dim)))
        self.coord_dim = min(coord_dim, max_cd)
        self.proj      = nn.Linear(2, self.coord_dim, bias=False)
        init_scale     = 1.0 / math.sqrt(max(feat_dim, 1))
        self.coord_scale = nn.Parameter(torch.tensor(init_scale))
        self._cache: dict = {}
    def _get_coords(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, str(device))
        if key not in self._cache:
            ys = torch.linspace(0, 1, H, device=device)
            xs = torch.linspace(0, 1, W, device=device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            self._cache[key] = torch.stack([gx, gy], dim=-1).reshape(H * W, 2)
        return self._cache[key]
    def forward(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        coords = self._get_coords(H, W, tokens.device)
        pe     = self.proj(coords) * self.coord_scale
        pe     = pe.unsqueeze(0).expand(tokens.size(0), -1, -1)
        return torch.cat([tokens, pe], dim=-1)
class NoiseAwareAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.Sigmoid())
        self.channel = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.Sigmoid())
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.spatial(h) * self.channel(h) * h
class DynamicGraphLearning(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, max_nodes: int = 256):
        super().__init__()
        self.d         = max(channels // reduction, 16)
        self.max_nodes = max_nodes
        self.proj      = nn.Conv2d(channels, self.d, 1, bias=False)
        self.back      = nn.Conv2d(self.d, channels, 1, bias=False)
        self.norm      = nn.BatchNorm2d(channels)
        self.attn_drop = nn.Dropout(p=0.1)
        self.alpha_raw = nn.Parameter(torch.zeros(1))
    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw) * 0.9 + 0.05
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        needs_pool  = H * W > self.max_nodes
        if needs_pool:
            side   = int(math.isqrt(self.max_nodes))
            x_pool = F.adaptive_avg_pool2d(x, (side, side))
        else:
            x_pool, side = x, H
        h      = self.proj(x_pool).flatten(2)
        h_norm = F.normalize(h, dim=1)
        A      = torch.bmm(h_norm.transpose(1, 2), h_norm)
        A      = F.softmax(A / math.sqrt(self.d), dim=-1)
        A      = self.attn_drop(A)
        msg    = torch.bmm(h, A).reshape(B, self.d, side, side)
        msg    = self.back(msg)
        if needs_pool:
            msg = F.interpolate(msg, size=(H, W), mode="bilinear", align_corners=False)
        return self.norm(x + self.alpha * msg)
class SparseGraphAttention(nn.Module):
    def __init__(self, in_dim: int, graph_dim: int = 128,
                 d_head: int = 32, k: int = 6, num_heads: int = 4,
                 dropout: float = 0.1, coord_dim: int = 2):
        super().__init__()
        total          = d_head * num_heads
        self.k         = k
        self.num_heads = num_heads
        self.d_head    = d_head
        self.scale     = math.sqrt(d_head)
        self.pos_enc  = PositionalEncoding2D(coord_dim=coord_dim, feat_dim=in_dim)
        actual_cdim   = self.pos_enc.coord_dim
        self.down = nn.Linear(in_dim + actual_cdim, graph_dim, bias=False)
        self.up   = nn.Linear(graph_dim, in_dim, bias=False)
        self.proj_q   = nn.Linear(graph_dim, total, bias=False)
        self.proj_k   = nn.Linear(graph_dim, total, bias=False)
        self.proj_v   = nn.Linear(graph_dim, total, bias=False)
        self.out_proj = nn.Linear(total, graph_dim, bias=False)
        self.attn_drop = nn.Dropout(p=dropout)
        self.ff_drop   = nn.Dropout(p=dropout)
        self.norm      = nn.LayerNorm(in_dim)
    @torch.no_grad()
    def _knn_indices(self, x: torch.Tensor) -> torch.Tensor:
        sq   = (x * x).sum(-1, keepdim=True)
        dist = sq + sq.transpose(1, 2) - 2.0 * torch.bmm(x, x.transpose(1, 2))
        dist.diagonal(dim1=1, dim2=2).fill_(1e9)
        return dist.topk(self.k, dim=-1, largest=False).indices
    @staticmethod
    def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        B, N, k = idx.shape
        D       = x.shape[-1]
        flat    = idx.reshape(B, -1)
        return (torch.gather(x, 1, flat.unsqueeze(-1).expand(-1, -1, D))
                .reshape(B, N, k, D))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        tokens    = x.flatten(2).transpose(1, 2)
        tokens_pe = self.pos_enc(tokens, H, W)
        h         = self.down(tokens_pe)
        q      = self.proj_q(h)
        k_feat = self.proj_k(h)
        v_feat = self.proj_v(h)
        knn_idx = self._knn_indices(h.detach())
        k_nbr   = self._gather(k_feat, knn_idx)
        v_nbr   = self._gather(v_feat, knn_idx)
        Hd, d = self.num_heads, self.d_head
        q     = q.view(B, N, Hd, d)
        k_nbr = k_nbr.view(B, N, self.k, Hd, d).permute(0, 1, 3, 2, 4)
        v_nbr = v_nbr.view(B, N, self.k, Hd, d).permute(0, 1, 3, 2, 4)
        attn = (q.unsqueeze(3) * k_nbr).sum(-1) / self.scale
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out  = (attn.unsqueeze(-1) * v_nbr).sum(-2)
        out  = self.ff_drop(self.out_proj(out.reshape(B, N, Hd * d)))
        out  = self.up(out)
        return self.norm(out + tokens).transpose(1, 2).reshape(B, C, H, W)
class GraphBottleneck(nn.Module):
    def __init__(self, channels: int, k: int = 6,
                 graph_dim: int = 128, d_head: int = 32, num_heads: int = 4,
                 dyn_max_nodes: int = 256, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.noise    = NoiseAwareAttention(channels)
        self.dyngraph = DynamicGraphLearning(channels, max_nodes=dyn_max_nodes)
        self.gat      = SparseGraphAttention(
            in_dim=channels, graph_dim=graph_dim,
            d_head=d_head, k=k, num_heads=num_heads)
        self.ff      = nn.Sequential(
            ConvBNReLU(channels, channels),
            ConvBNReLU(channels, channels))
        self.ff_norm = nn.BatchNorm2d(channels)
        self.ff_drop = nn.Dropout2d(p=0.1)
    def _gat_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.gat(x)
    def _ff_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_norm(self.ff_drop(self.ff(x)) + x)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.noise(x)
        x = self.dyngraph(x)
        if self.use_checkpoint and x.requires_grad:
            x = grad_ckpt.checkpoint(self._gat_fn, x, use_reentrant=False)
            x = grad_ckpt.checkpoint(self._ff_fn,  x, use_reentrant=False)
        else:
            x = self._gat_fn(x)
            x = self._ff_fn(x)
        return x
class AttentionGate(nn.Module):
    def __init__(self, x_ch: int, g_ch: int, inter_ch: Optional[int] = None):
        super().__init__()
        inter_ch = inter_ch or max(x_ch // 2, 1)
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.psi(F.relu(self.W_x(x) + self.W_g(g), inplace=True)) * x
class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_dw: bool = False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.ag   = AttentionGate(x_ch=skip_ch, g_ch=out_ch)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch, use_dw),
            ConvBNReLU(out_ch, out_ch, use_dw))
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        skip = self.ag(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))
class AttentionPoolingHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int = 64, use_attention: bool = True):
        super().__init__()
        self.use_att = use_attention
        if use_attention:
            self.att_w = nn.Conv2d(in_ch, 1, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.use_att:
            w      = self.att_w(x).reshape(B, 1, -1)
            w      = F.softmax(w, dim=-1)
            pooled = (w * x.reshape(B, C, -1)).sum(-1)
        else:
            pooled = x.mean(dim=(2, 3))
        return self.fc(pooled)
class MCDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.p, training=True)
class AoPGAUNet(nn.Module):
    def __init__(
        self,
        in_channels:    int  = 3,
        num_classes:    int  = 2,
        base_ch:        int  = 32,
        k_neighbors:    int  = 6,
        graph_dim:      int  = 128,
        d_head:         int  = 32,
        num_heads:      int  = 4,
        use_dw:         bool = True,
        use_att_pool:   bool = True,
        use_mc_dropout: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        c     = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        bn_ch = c[3] * 2
        self.enc1 = EncoderBlock(in_channels, c[0], use_dw)
        self.enc2 = EncoderBlock(c[0], c[1], use_dw)
        self.enc3 = EncoderBlock(c[1], c[2], use_dw)
        self.enc4 = EncoderBlock(c[2], c[3], use_dw)
        self.bottleneck_conv = nn.Sequential(
            ConvBNReLU(c[3], bn_ch, use_dw),
            ConvBNReLU(bn_ch, bn_ch, use_dw))
        self.graph_bottleneck = GraphBottleneck(
            channels=bn_ch, k=k_neighbors,
            graph_dim=min(graph_dim, bn_ch), d_head=d_head,
            num_heads=num_heads, dyn_max_nodes=256,
            use_checkpoint=use_checkpoint)
        self.mc_drop = MCDropout(p=0.1) if use_mc_dropout else nn.Identity()
        self.dec1 = DecoderBlock(bn_ch, c[3], c[3], use_dw)
        self.dec2 = DecoderBlock(c[3],  c[2], c[2], use_dw)
        self.dec3 = DecoderBlock(c[2],  c[1], c[1], use_dw)
        self.dec4 = DecoderBlock(c[1],  c[0], c[0], use_dw)
        self.seg_head = nn.Conv2d(c[0], num_classes, 1)
        self.aop_head = AttentionPoolingHead(in_ch=c[0], hidden=64,
                                             use_attention=use_att_pool)
        self._init_weights()
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _dbg_shape("input", x)
        x, s1 = self.enc1(x);  _dbg_shape("enc1 skip", s1)
        x, s2 = self.enc2(x);  _dbg_shape("enc2 skip", s2)
        x, s3 = self.enc3(x);  _dbg_shape("enc3 skip", s3)
        x, s4 = self.enc4(x);  _dbg_shape("enc4 skip", s4)
        x = self.bottleneck_conv(x);   _dbg_shape("bottleneck_conv", x)
        x = self.graph_bottleneck(x);  _dbg_shape("graph_bottleneck", x)
        x = self.mc_drop(x)
        x = self.dec1(x, s4);  _dbg_shape("dec1", x)
        x = self.dec2(x, s3);  _dbg_shape("dec2", x)
        x = self.dec3(x, s2);  _dbg_shape("dec3", x)
        x = self.dec4(x, s1);  _dbg_shape("dec4 (pre-head)", x)
        seg   = self.seg_head(x)
        angle = self.aop_head(x) * 180.0
        _dbg_shape("seg_logits", seg)
        _dbg_shape("aop_angle",  angle)
        mark_shapes_done()
        return seg, angle
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        nc     = logits.shape[1]
        probs  = F.softmax(logits, dim=1)
        tgt_oh = F.one_hot(targets.long(), nc).permute(0, 3, 1, 2).float()
        inter  = (probs * tgt_oh).sum(dim=(2, 3))
        union  = probs.sum(dim=(2, 3)) + tgt_oh.sum(dim=(2, 3))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets.long(), reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("laplacian", lap.view(1, 1, 3, 3))
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)[:, 1:2]
        tgt   = targets.float().unsqueeze(1)
        pe    = F.conv2d(probs, self.laplacian, padding=1).abs()
        te    = F.conv2d(tgt,   self.laplacian, padding=1).abs()
        return F.mse_loss(pe, te)
class CombinedLoss(nn.Module):
    def __init__(self, dice_w: float = 1.0, bnd_w: float = 0.5,
                 ce_w: float = 1.0, aop_w: float = 0.2,
                 use_focal: bool = False,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.dice   = DiceLoss()
        self.bnd    = BoundaryLoss()
        self.ce     = FocalLoss() if use_focal else nn.CrossEntropyLoss(weight=pos_weight)
        self.dice_w = dice_w
        self.bnd_w  = bnd_w
        self.ce_w   = ce_w
        self.aop_w  = aop_w
    def forward(self, seg_logits: torch.Tensor, seg_targets: torch.Tensor,
                aop_pred: torch.Tensor,
                aop_targets: Optional[torch.Tensor]) -> torch.Tensor:
        loss = (self.dice_w * self.dice(seg_logits, seg_targets)
                + self.ce_w  * self.ce(seg_logits,  seg_targets)
                + self.bnd_w * self.bnd(seg_logits, seg_targets))
        if aop_targets is not None:
            loss = loss + self.aop_w * F.mse_loss(
                aop_pred.squeeze(1), aop_targets.float() * 180.0)
        return loss
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def build_model(
    in_channels:    int  = 3,
    num_classes:    int  = 2,
    base_ch:        int  = 32,
    use_dw:         bool = True,
    use_att_pool:   bool = True,
    use_mc_dropout: bool = False,
    use_checkpoint: bool = False,
) -> AoPGAUNet:
    """Factory with RTX 2050-safe defaults."""
    model = AoPGAUNet(
        in_channels=in_channels, num_classes=num_classes, base_ch=base_ch,
        k_neighbors=6, graph_dim=128, d_head=32, num_heads=4,
        use_dw=use_dw, use_att_pool=use_att_pool,
        use_mc_dropout=use_mc_dropout, use_checkpoint=use_checkpoint)
    print(f"[AoPGAUNet] Parameters: {count_parameters(model):,}")
    return model
if __name__ == "__main__":
    import model_optimized as _mod
    _mod.DEBUG = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(in_channels=3, num_classes=2, base_ch=32,
                         use_dw=True, use_att_pool=True).to(device)
    x      = torch.randn(2, 3, 256, 256, device=device)
    amp_on = device.type == "cuda"
    with torch.amp.autocast(device_type=device.type, enabled=amp_on):
        seg, angle = model(x)
    assert seg.shape   == (2, 2, 256, 256)
    assert angle.shape == (2, 1)
    assert 0.0 <= angle.min().item() and angle.max().item() <= 180.0
    print(f"\nSeg  : {seg.shape}")
    print(f"Angle: {angle.shape}  [{angle.min().item():.1f}, {angle.max().item():.1f}]")
    _dbg_mem("post-forward")
    print("Validation passed.")
