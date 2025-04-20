import math
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import cv2
import numpy as np

_transform = Compose([
        Resize(
            (224,224), interpolation=InterpolationMode.BICUBIC
        ),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def imgprocess(img, patch_size=[14, 14], scale_factor=1):
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph

    ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
    img = ResizeOp(img).convert("RGB")
    return _transform(img)

def visualize(map, raw_image, resize):
    image = np.asarray(raw_image.copy())
    map = resize(map.unsqueeze(0))[0].cpu().numpy()
    color = cv2.applyColorMap((map*255).astype(np.uint8), cv2.COLORMAP_JET) # cv2 to plt
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    c_ret = np.clip(image * (1 - 0.5) + color * 0.5, 0, 255).astype(np.uint8)
    return c_ret

class Grad_ECLIP(object):
    def __init__(self, 
                 model,
                 text_embedding):
        super(Grad_ECLIP, self).__init__()
        self.clipmodel = model
        self.clip_inres = self.clipmodel.visual.input_resolution
        self.clip_ksize = self.clipmodel.visual.conv1.kernel_size
        
        self.text_embedding = text_embedding
        
    def attention_layer(self, q, k, v, num_heads=1):
        "Compute 'Scaled Dot Product Attention'"
        tgt_len, bsz, embed_dim = q.shape
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_heads = torch.bmm(attn_output_weights, v)
        assert list(attn_output_heads.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
        attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    
    def clip_encode_dense(self, x,n):
        vision_width = self.clipmodel.visual.transformer.width
        vision_heads = vision_width // 64
        print("[vision_width and vision_heads]:", vision_width, vision_heads)
        
        # modified from CLIP
        x = x.half()
        x = self.clipmodel.visual.conv1(x)  
        feah, feaw = x.shape[-2:]

        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        class_embedding = self.clipmodel.visual.class_embedding.to(x.dtype)
        x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

        ## scale position embedding as the image w-h ratio
        pos_embedding = self.clipmodel.visual.positional_embedding.to(x.dtype)
        tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
        pos_h = self.clip_inres // self.clip_ksize[0]
        pos_w = self.clip_inres // self.clip_ksize[1]
        assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
        img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
        print("[POS shape]:", img_pos.shape, (feah, feaw))
        img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
        img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
        pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
        x = x + pos_embedding
        x = self.clipmodel.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = torch.nn.Sequential(*self.clipmodel.visual.transformer.resblocks[:-n])(x)

        attns = []
        atten_outs = []
        vs = []
        qs = []
        ks = []
        for TR in self.clipmodel.visual.transformer.resblocks[-n:]:
            x_in = x
            x = TR.ln_1(x_in)
            linear = torch._C._nn.linear    
            q, k, v = linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
            attn_output, attn = self.attention_layer(q, k, v, 1)  # vision_heads=1
            attns.append(attn)
            atten_outs.append(attn_output)
            vs.append(v)
            qs.append(q)
            ks.append(k)
            
            x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)       
            x = x_after_attn + x_in
            x = x + TR.mlp(TR.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.visual.ln_post(x)
        x = x @ self.clipmodel.visual.proj
        return x, x_in, vs, qs, ks, attns, atten_outs, (feah, feaw)
    
    def sim_qk(self, q, k):
        q_cls = F.normalize(q[:1,0,:], dim=-1) 
        k_patch = F.normalize(k[1:,0,:], dim=-1)

        cosine_qk = (q_cls * k_patch).sum(-1) 
        cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
        cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
        cosine_qk = (cosine_qk-cosine_qk_min) / (cosine_qk_max-cosine_qk_min)
        return cosine_qk
    
    def grad_eclip(self, c, qs, ks, vs, attn_outputs, map_size):
        ## gradient on last attention output
        tmp_maps = []
        for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
            grad = torch.autograd.grad(
                c,
                attn_output,
                retain_graph=True)[0]

            grad_cls = grad[:1,0,:]
            v_patch = v[1:,0,:]
            cosine_qk = self.sim_qk(q, k).reshape(-1)
            tmp_maps.append((grad_cls * v_patch * cosine_qk[:,None]).sum(-1)) 
            
            del grad
            del grad_cls
            torch.cuda.empty_cache()
        emap = F.relu_(torch.stack(tmp_maps, dim=0)).sum(0)
        
        del tmp_maps
        torch.cuda.empty_cache()
        
        return emap.reshape(*map_size)
    
    def self_attn(self, attns, map_size):
        attn_patch = attns[-1][0,:1,1:].reshape(*map_size)
        print("[attn of cls token on lastv]:", attn_patch.shape)
        return attn_patch
    
    def __call__(self, img_preprocessed, id = None):
        """
        img_preprocessed    (1, c, w, h)
        """
        self.clipmodel.zero_grad()
        
        outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = self.clip_encode_dense(img_preprocessed, n=1)
        
        img_embedding = F.normalize(outputs[:,0], dim=-1)
        
        cosine = (img_embedding @ self.text_embedding.T)[0]
        
        map = self.grad_eclip(cosine[id], qs, ks, vs, atten_outs, map_size).detach().cpu().numpy()
        
        map -= map.min()
        map /= map.max()
        
        self.clipmodel.zero_grad()
        
        return map
        
        
        