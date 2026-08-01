import math
import numbers
import warnings

import torch
import torch.nn as nn
from einops import rearrange, repeat

try:
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    import selective_scan_cuda_core as selective_scan_cuda
    print("Using \"selective_scan_cuda_core\"")
except Exception as e:
    warnings.warn(f"{e}\n\"selective_scan_cuda_core\" not found, use default \"selective_scan_cuda\" instead.")
    # print(e, flush=True)
    SSMODE = "mamba_ssm"
    import selective_scan_cuda


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        softmax_version=False,
        forward_type="v2",
        # ======================
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        #d_model model dim
        d_expand = int(ssm_ratio * d_model)
        #d_inner  dim in model, for channel it should be 2
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.softmax_version = softmax_version
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv

        #cccc
        dc_inner = 4 
        self.dtc_rank = 6 #6
        self.dc_state = 16 #16
        #self.conv_innerc =  nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        #self.conv_innerc = nn.Linear(1, dc_inner, bias=bias, **factory_kwargs)
        self.conv_cin = nn.Conv2d(in_channels=1, out_channels=dc_inner, kernel_size=1, stride=1, padding=0)
        self.conv_cout = nn.Conv2d(in_channels=dc_inner, out_channels=1, kernel_size=1, stride=1, padding=0)
        #self.conv_outc = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.forward_core=self.forward_corev1

        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1        
        self.KC = 2
        self.K2C = self.KC if forward_type not in ["share_a"] else 1

        self.cforward_core = self.cforward_corev1
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.channel_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')
        # self.channel_norm_v2 = LayerNorm(d_state, LayerNorm_type='WithBias')

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.in_conv = nn.Conv2d(in_channels=d_model, out_channels=d_expand * 2, kernel_size=1, stride=1, padding=0)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )


        self.out_norm = LayerNorm(d_inner, LayerNorm_type='WithBias')

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        # xc proj ============================
        self.xc_proj = [
            nn.Linear(dc_inner, (self.dtc_rank + self.dc_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.KC)
        ]
        self.xc_proj_weight = nn.Parameter(torch.stack([tc.weight for tc in self.xc_proj], dim=0)) # (K, N, inner)
        del self.xc_proj


        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)

        # out proj =======================================
        self.out_conv = nn.Conv2d(in_channels=d_expand, out_channels=d_model, kernel_size=1, stride=1, padding=0)
        # self.out_conv_v2 = nn.Conv2d(in_channels=d_state, out_channels=d_state, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Dsc = nn.Parameter(torch.ones((self.K2C * dc_inner)))
        self.Ac_logs = nn.Parameter(torch.randn((self.K2C * dc_inner, self.dc_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dtc_projs_weight = nn.Parameter(torch.randn((self.KC, dc_inner, self.dtc_rank)).contiguous())
        self.dtc_projs_bias = nn.Parameter(torch.randn((self.KC, dc_inner))) 
        self.conv_layers = {}

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj


    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log


    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    

    
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W

        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1) #一个h,w展开，一个w,h展开，然后堆在一起
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l) #把上面那俩再翻译下，然后堆在一起
            return xs 
        
        #四个方向
        if self.K == 4:
            # K = 4
            xs = cross_scan_2d(x) # (b, k, d, l) #[batch_size, 4, channels, height * width]

            #print("x shape", x.shape) # 8,96,128,128
            #print("xs shape", xs.shape) # 8,4,96, 16384
            #print("Ac_logs shape", self.A_logs.shape) #384,16
            #print("dtc_projs_weight shape", self.dt_projs_weight.shape) #4,96,6
            #print("xc_proj_weight shape", self.x_proj_weight.shape) #4,38,96
            #print("Dsc shape", self.Ds.shape) # 384
            #print("dtc_projs_bias shape", self.dt_projs_bias.shape) #4,96

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L) # (b, k * d, l)
            dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
            As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
            Ds = self.Ds # (k * d)
            dt_projs_bias = self.dt_projs_bias.view(-1) # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
            # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

            #print("xs shape", xs.shape) #8,768,1024
            #print("dts shape", dts.shape) #8, 768, 1024
            #print("As shape", As.shape)#768, 16
            #print("Bs shape", Bs.shape)#8, 4, 16, 1024
            #print("Cs shape", Cs.shape)#8, 4, 16, 1024
            #print("Ds shape", Ds.shape)#768
            #print("dt_projs_bias shape", dt_projs_bias.shape) #768
            out_y = self.selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, 4, -1, L)
            # assert out_y.dtype == torch.float16


        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        

        y = y.view(B, C, H, W)
        y = self.out_norm(y).to(x.dtype)
        
        return y

    def cforward_corev1(self, xc: torch.Tensor):
        self.selective_scanC = selective_scan_fn_v1

        b,d,h,w = xc.shape
        
        #xc = self.pooling(xc).squeeze(-1).permute(0,2,1).contiguous() #b,1,d, >1!
        #print("xc shape", xc.shape) # 8,1,96
        xc = self.pooling(xc) #b,d,1,1
        xc = xc.permute(0,2,1,3).contiguous() #b,1,d,1
        xc = self.conv_cin(xc) #b,4,d,1
        xc = xc.squeeze(-1) #b,4,d

        #xc = xc.permute(0,2,1).contiguous()
        #xc = self.conv_innerc(xc)
        #xc = xc.permute(0,2,1).contiguous()

        B, D, L = xc.shape #b,1,c
        D, N = self.Ac_logs.shape #2,16
        K, D, R = self.dtc_projs_weight.shape #2,1,6

        #print("Ac_logs shape", self.Ac_logs.shape) #2,16
        #print("dtc_projs_weight shape", self.dtc_projs_weight.shape) #2,1,6
        #print("xc_proj_weight shape", self.xc_proj_weight.shape) #2,38,1
        #print("Dsc shape", self.Dsc.shape) # 2
        #print("dtc_projs_bias shape", self.dtc_projs_bias.shape) #2,1

        xsc = torch.stack([xc, torch.flip(xc, dims=[-1])], dim=1) #input:b,d,l output:b,2,d,l
        #print("xsc shape", xsc.shape) # 8,2,1,96

        xc_dbl = torch.einsum("b k d l, k c d -> b k c l", xsc, self.xc_proj_weight) #8,2,1,96; 2,38,1 ->8,2,38,96
        
        dts, Bs, Cs = torch.split(xc_dbl, [self.dtc_rank, self.dc_state, self.dc_state], dim=2) # 8,2,38,96-> 6,16,16
        #dts:8,2,6,96 bs,cs:8,2,16,96
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dtc_projs_weight).contiguous()

        xsc = xsc.view(B, -1, L) # (b, k * d, l) 8,2,96
        dts = dts.contiguous().view(B, -1, L).contiguous() # (b, k * d, l) 8,2,96
        As = -torch.exp(self.Ac_logs.float())  # (k * d, d_state) 2,16
        Ds = self.Dsc # (k * d) 2 
        dt_projs_bias = self.dtc_projs_bias.view(-1) # (k * d)2

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16
        

        #print("channel xs shape", xsc.shape) #8, 2, 192
        #print("channel dts shape", dts.shape)#8, 2, 192
        #print("channel As shape", As.shape)#2, 16
        #print("channel Bs shape", Bs.shape)#8, 2, 16, 192
        #print("channel Cs shape", Cs.shape)#8, 2, 16, 192
        #print("channel Ds shape", Ds.shape)#2
        #print("channel dt_projs_bias shape", dt_projs_bias.shape) #2
        out_y = self.selective_scanC(
            xsc, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, 2, -1, L)

        y = out_y[:, 0].float() + torch.flip(out_y[:, 1], dims=[-1]).float()
        #y = xsc[:, 0].float() + torch.flip(xsc[:, 1], dims=[-1]).float()


        #y: b,4,d
        y = y.unsqueeze(-1) # b,4,d,1
        y = self.conv_cout(y) # b,1,d,1
        y = y.transpose(dim0=1, dim1=2).contiguous() # b,d,1,1
        # if y.shape[1] == self.d_state:
        #     y = self.channel_norm_v2(y)
        # else:
        y = self.channel_norm(y)
        y = y.to(xc.dtype)

        #y = y.transpose(dim0=1, dim1=2).contiguous().unsqueeze(-1).contiguous()
        #y = self.channel_norm(y)
        #y = y.to(xc.dtype)

        #y = y.permute(0,2,1,3).contiguous()
        #y = self.conv_outc(y)
        #y = y.permute(0,2,1,3).contiguous()
        
        return y

    def get_conv1d_layer(self, in_channels, device):
        if in_channels not in self.conv_layers:
            self.conv_layers[in_channels] = nn.Conv1d(in_channels=in_channels,
                                                      out_channels=1,
                                                      kernel_size=1,
                                                      stride=1).to(device)
        return self.conv_layers[in_channels]
    
    def reduce_channel(self, x):
        b, d, h, w, c = x.shape
        conv1d = self.get_conv1d_layer(c, x.device)
        x = x.contiguous().view(-1, c, 1)
        x = conv1d(x).view(b, d, h, w)
        return x


    # def forward(self, x: torch.Tensor, **kwargs): #v1
    #     #input: b,d,h,w
    #     #output: b,d,h,w
    #     ############# I added ################################
    #     B, C, D, H, W = x.shape
    #     residual = x
    #     x = rearrange(x, 'b c d h w -> b h w c d')

    #     xz = self.in_proj(x)
        

    #     # xz = self.in_conv(x)

    #     if self.d_conv > 1:
    #         x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
    #         # I added
    #         x = rearrange(x, 'b h w c d -> b h w d c').mean(-1)
    #         z = rearrange(z, 'b h w c d -> b h w d c').mean(-1)
    #         # --------
    #         z = self.act(z)
    #         # I added
    #         z = x.permute(0, 3, 1, 2).contiguous()
    #         # --------

    #         x = x.permute(0, 3, 1, 2).contiguous()
    #         x = self.act(self.conv2d(x)) # (b, d, h, w)
    #     else:
    #         xz = self.act(xz)
    #         x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
    #     ###############################################################
        
    #     # x, z = xz.chunk(2, dim=1) # (b, d, h, w)
    #     # if not self.softmax_version:
    #     #     z = self.act(z)
    #     # x = self.act(self.conv2d(x)) # (b, d, h, w)
            
        
    #     y1 = self.forward_core(x)
    #     y2 = y1 * z
    #     c = self.cforward_core(y2)#x:b,d,h,w; output:b,d,1,1
    #     y3 = y2 * c
    #     y2 = y3 + y2
    #     out = self.out_conv(y2)

    #     #############
    #     out = out.permute(0, 2, 3, 1)
    #     out = out.unsqueeze(-2)
    #     out = rearrange(out, 'b h w c d -> b c d h w')
    #     out = out + residual
    #     #############
    #     # print(x.shape, z.shape, y1.shape, y2.shape, out.shape); exit()
    #     return out
    
    def forward(self, x: torch.Tensor, **kwargs): # v2
        #input: b,d,h,w
        #output: b,d,h,w
        ############# I added ################################
        B, C, D, H, W = x.shape
        residual = x
        x = rearrange(x, 'b c d h w -> b h w c d')

        xz = self.in_proj(x)
        

        # xz = self.in_conv(x)

        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            # I added
            # x = rearrange(x, 'b h w c d -> b h w d c').mean(-1)
            x = rearrange(x, 'b h w c d -> b h w d c')
            x = self.reduce_channel(x)
            # --------
            z = self.act(z)
            # # I added
            # z = x.permute(0, 3, 1, 2).contiguous()
            # # --------

            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
        else:
            xz = self.act(xz)
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        ###############################################################
        
        # x, z = xz.chunk(2, dim=1) # (b, d, h, w)
        # if not self.softmax_version:
        #     z = self.act(z)
        # x = self.act(self.conv2d(x)) # (b, d, h, w)
            
        
        y1 = self.forward_core(x)
        y1 = y1.permute(0, 2, 3, 1)  # b, h, w, d
        y1 = y1.unsqueeze(-2)
        y2 = y1 * z
        # y2 = rearrange(y2, 'b h w c d -> b d h w c').mean(-1)
        y2 = rearrange(y2, 'b h w c d -> b d h w c')
        y2 = self.reduce_channel(y2)
        c = self.cforward_core(y2)#x:b,d,h,w; output:b,d,1,1
        y3 = y2 * c
        y2 = y3 + y2
        out = self.out_conv(y2)
        # print(f"X: {x.shape}\nZ: {z.shape}\nY1: {y1.shape}\nY2: {y2.shape}")
        # print(f"c: {c.shape}\ny3: {y3.shape}")
        # exit()
        
        #############
        out = out.permute(0, 2, 3, 1)
        out = out.unsqueeze(-2)
        out = rearrange(out, 'b h w c d -> b c d h w')
        out = out + residual
        #############
        # print(x.shape, z.shape, y1.shape, y2.shape, out.shape); exit()
        return out

    # def forward(self, x: torch.Tensor, **kwargs): # v3
    #     #input: b,d,h,w
    #     #output: b,d,h,w
    #     ############# I added ################################
    #     B, C, D, H, W = x.shape
    #     residual = x
    #     x = rearrange(x, 'b c d h w -> b h w c d')

    #     xz = self.in_proj(x)

    #     if self.d_conv > 1:
    #         x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
    #         # I added
    #         x = rearrange(x, 'b h w c d -> b h w d c').mean(-1)
    #         # --------
    #         z = self.act(z)

    #         x = x.permute(0, 3, 1, 2).contiguous()
    #         x = self.act(self.conv2d(x)) # (b, d, h, w)
    #     else:
    #         xz = self.act(xz)
    #         x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
            
        
    #     y1 = self.forward_core(x)
    #     y1 = y1.permute(0, 2, 3, 1)  # b, h, w, d
    #     y1 = y1.unsqueeze(-2)
    #     y2 = y1 * z
        
    #     y2_d = rearrange(y2, 'b h w c d -> b d h w c').mean(-1) ; print("Y2_D:", y2_d.shape)
    #     d = self.cforward_core(y2_d)#x:b,d,h,w; output:b,d,1,1
    #     y3_d = y2_d * d  # b,d,h,w
    #     y2_d_merged = y3_d + y2_d 
    #     out1 = self.out_conv(y2_d_merged)

    #     y2_c = rearrange(y2, 'b h w c d -> b c h w d').mean(-1) ; print("Y2_C:", y2_c.shape)
    #     c = self.cforward_core(y2_c)#x:b,d,h,w; output:b,c,1,1
    #     y3_c = y2_c * c # b,c,h,w
    #     y2_c_merged = y3_c + y2_c 
    #     out2 = self.out_conv_v2(y2_c_merged)

    #     # print(f"X: {x.shape}\nZ: {z.shape}\nY1: {y1.shape}\nY2: {y2.shape}")
    #     # print(f"c: {c.shape}\ny3: {y3.shape}")
    #     # exit()
        
    #     #############
    #     out1 = out1.permute(0, 2, 3, 1) # b, h, w, d
    #     out1 = out1.unsqueeze(-2) # b,h,w,c,d

    #     out2 = out2.permute(0, 2, 3, 1) # b, h, w, c
    #     out2 = out2.unsqueeze(-1) # b,h,w,c,d

    #     out = out1 + out2

    #     out = rearrange(out, 'b h w c d -> b c d h w')
    #     out = out + residual
    #     #############
    #     # print(x.shape, z.shape, y1.shape, y2.shape, out.shape); exit()
    #     return out

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


######################################################################

class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        # input_t: float, fp16, bf16; weight_t: float;
        # u, B, C, delta: input_t
        # D, delta_bias: float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        if D is not None and (D.dtype != torch.float):
            ctx._d_dtype = D.dtype
            D = D.float()
        if delta_bias is not None and (delta_bias.dtype != torch.float):
            ctx._delta_bias_dtype = delta_bias.dtype
            delta_bias = delta_bias.float()
        
        assert u.shape[1] % (B.shape[1] * nrows) == 0 
        assert nrows in [1, 2, 3, 4] # 8+ is too slow to compile

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
        )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        
        _dD = None
        if D is not None:
            if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                _dD = dD.to(ctx._d_dtype)
            else:
                _dD = dD

        _ddelta_bias = None
        if delta_bias is not None:
            if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
            else:
                _ddelta_bias = ddelta_bias

        return (du, ddelta, dA, dB, dC, _dD, _ddelta_bias, None, None)


def selective_scan_fn_v1(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False, with_Group=True)
    return flops

