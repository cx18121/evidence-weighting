"""One-model-at-a-time exact readout and forward-hook access (no offloading)."""
from contextlib import contextmanager
from pathlib import Path
import torch
import torch.nn as nn
from prompts import QUESTION, NUMERIC_QUESTION


def load_model(name, device='cuda', dtype='bfloat16'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    dev = torch.device(device)
    if name.startswith('jaredpalmer/kev-') or (Path(name) / 'head.pt').exists():
        from kev.checkpoint import load, LoadOptions
        tok, model = load(name, dev, LoadOptions(dtype=getattr(torch, dtype), backend='torch',
                                                  cuda_graphs=False, fused=False, merge=False))
        return tok, model, True
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    kw = {'torch_dtype': getattr(torch, dtype), 'low_cpu_mem_usage': True}
    if name.startswith('mistralai/Mistral-Small-3.2-24B'):
        from transformers import BitsAndBytesConfig
        if device != 'cuda':
            raise RuntimeError('Mistral 8-bit requires CUDA')
        kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        kw['device_map'] = 'auto'
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    if 'device_map' not in kw:
        model = model.to(dev)
    return tok, model, False


def layers_of(model, kev=False):
    base = model.lm if kev else model
    found = [(name, module) for name, module in base.named_modules()
             if isinstance(module, nn.ModuleList) and name.endswith('layers') and len(module) > 2]
    if not found:
        raise RuntimeError('Cannot identify transformer layers; inspect model.named_modules()')
    return max(found, key=lambda p: len(p[1]))[1]


def answer_token_ids(tok):
    # Conditional P(Yes | {Yes, No}) is *not* total semantic yes mass; the specified readout
    # assumes both answer spellings start with distinct single tokens under this tokenizer.
    yes, no = [tok.encode(s, add_special_tokens=False) for s in ('Yes', 'No')]
    if len(yes) != 1 or len(no) != 1 or yes == no:
        raise ValueError(f'Yes/No not single distinct tokens: {yes}, {no}')
    return yes[0], no[0]


def lm_inputs(tok, case, numeric=False):
    messages = [{'role': 'user', 'content': case['state'] + '\n\n' + (NUMERIC_QUESTION if numeric else QUESTION)}]
    kwargs = {'add_generation_prompt': True}
    if 'qwen3' in str(getattr(tok, 'name_or_path', '')).lower():
        kwargs['enable_thinking'] = False
    # Render the chat template to text before tokenizing.
    text = tok.apply_chat_template(messages, tokenize=False, **kwargs)
    ids = tok(text, add_special_tokens=False)['input_ids']
    return torch.tensor([ids], dtype=torch.long)


def kev_enc(tok, model, case):
    # Exactly the repo's Noul question/option path; [no, yes] ordering from kev.api.to_record.
    from kev.api import SystemOneRequest, to_record
    req = SystemOneRequest.model_validate({'state': case['state'], 'questions': {
        'answer': {'type': 'noul', 'instructions': QUESTION}}})
    rec, _ = to_record(req)
    return model.encode(tok, rec, strict=True, max_state=65536, max_branch=73728)


@torch.inference_mode()
def score(tok, model, kev, case, numeric=False):
    if kev:
        if numeric:
            raise ValueError('Kev has a Noul head, not next-token numeric output')
        enc = kev_enc(tok, model, case)
        z = model.forward(enc)[0].float()
        return float(z.softmax(-1)[1].item())
    ids = lm_inputs(tok, case, numeric).to(next(model.parameters()).device)
    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    if not numeric:
        yes, no = answer_token_ids(tok)
        return float(torch.softmax(logits[torch.tensor([yes, no], device=logits.device)], -1)[0])
    # Restricted first-token numerical distribution, valid ONLY if each integer 0..100 is one token.
    toks = [tok.encode(str(i), add_special_tokens=False) for i in range(101)]
    if not all(len(x) == 1 for x in toks) or len({x[0] for x in toks}) != 101:
        raise ValueError('numeric readout unavailable: integers 0..100 not unique single tokens')
    weights = torch.softmax(logits[torch.tensor([x[0] for x in toks], device=logits.device)], -1)
    return float((weights * torch.arange(101, device=weights.device)).sum() / 100)


def _first(out):
    return out[0] if isinstance(out, tuple) else out


def _replace(out, t):
    return (t,) + out[1:] if isinstance(out, tuple) else t


@contextmanager
def layer_hook(model, kev, layer, kind, callback):
    """callback(output vector) -> replacement or None. layer=0 captures embedding input to first layer;
    layer i>0 captures layer i-1 output (unnormalized residual); attn at layer i>0 captures its attention output.
    No activation edits to other positions. Recompute with full forward for hybrid DeltaNet correctness.
    """
    layers = layers_of(model, kev)
    if kind == 'residual' and layer == 0:
        def pre(_module, args):
            x = args[0]
            y = callback(x[0, -1].detach().float())
            if y is None: return None
            x = x.clone(); x[0, -1] = y.to(x.dtype)
            return (x,) + args[1:]
        h = layers[0].register_forward_pre_hook(pre)
    else:
        if layer < 1 or layer > len(layers): raise ValueError('layer out of range')
        if kind == 'residual':
            mod = layers[layer - 1]
        elif kind == 'attention':
            mod = getattr(layers[layer - 1], 'self_attn', None)
            if mod is None: raise RuntimeError(f'layer {layer} has no self_attn (hybrid linear layer)')
        else: raise ValueError(kind)
        def post(_module, _args, out):
            t = _first(out)
            y = callback(t[0, -1].detach().float())
            if y is None: return None
            t = t.clone(); t[0, -1] = y.to(t.dtype)
            return _replace(out, t)
        h = mod.register_forward_hook(post)
    try: yield
    finally: h.remove()
