<div align=center>
<img width="50%" alt="image" src="https://github.com/user-attachments/assets/f801617b-8959-474a-a565-6b8897e2fcbf" />
<h1 align="center">Defuser 🔧</h1>
</div>

<p align="center">
    <a href="https://github.com/ModelCloud/Defuser/releases" style="text-decoration:none;"><img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/Defuser.svg"></a>
    <a href="https://pypi.org/project/Defuser/" style="text-decoration:none;"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/Defuser"></a>
    <a href="https://pepy.tech/projects/Defuser" style="text-decoration:none;"><img src="https://static.pepy.tech/badge/Defuser" alt="PyPI Downloads"></a>
    <a href="https://github.com/ModelCloud/Defuser/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/Defuser"></a>
    <a href="https://huggingface.co/modelcloud/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ModelCloud-%23ff8811.svg"></a>
</p>

🧩 Defuser converts select Hugging Face Transformers `5.3.0+` fused or stacked MoE and MLP blocks back into plain, per-expert `nn.Linear` modules. It keeps the forward math intact while exposing individual projections again so quantizers, activation capture, debugging hooks, and checkpoint tooling can work against a simple module layout instead of fused expert tensors.

✅ Defuser is designed and CI-tested for `transformers>=5.3.0`, and support is only offered for that version range.

## 🎯 Purpose

Defuser exists for cases where newer Transformers modeling code optimizes model structure in ways that are good for runtime, but harder for tooling that needs direct access to individual projections.

Depending on the model family, Defuser can:

- 🧵 patch a supported model class before load so HF instantiates a defused block directly
- ✂️ split fused tensors such as `gate_up_proj` into `gate_proj` + `up_proj`
- 🧱 convert 3D expert tensors, including registered expert buffers, into numbered expert `nn.Linear` modules
- 🧮 preserve the original fused math while presenting a naive module structure again

🛠️ Public API:

```python
from defuser import convert_model, replace_fused_blocks
```

- 🧰 `replace_fused_blocks(model_type)` patches supported HF model classes before `from_pretrained()` or direct model construction.
- 🔄 `convert_model(model, cleanup_original=False, max_layers=None, filter=None)` converts an already loaded model in place. This is the runtime defusion path for supported post-load expert and MLP conversions, including `qwen3_5_moe` style checkpoints.
- 🧪 Defuser is designed and CI-tested for `transformers>=5.3.0`, and support is only offered for that version range. Older versions log a warning on these public APIs and are skipped as unsupported.
- 🧭 Some model families appear in both support tables. Full models can be prepatched with `replace_fused_blocks(...)`, while standalone fused expert modules from those same families can still be runtime-defused with `convert_model(...)`.

`filter` is an optional list of PCRE regex rules evaluated against full module paths such as `model.layers.0.mlp.experts`:

- ✅ `+:regex` explicitly includes matching candidate module paths
- 🚫 `-:regex` explicitly excludes matching candidate module paths
- ➕ `regex` is shorthand for `+:regex`
- 🛡️ negative rules take priority over positive rules
- 🎯 when `filter` is provided, a candidate module is defused only if it matches at least one positive rule and no negative rules

## ✅ Supported Models

Defuser currently supports the following `transformers>=5.3.0` `model_type` values.

### 🧰 `replace_fused_blocks(model_type)` before load

| Model type | Defused op performed ⚙️ |
| --- | --- |
| `glm4_moe` | Replaces `Glm4MoeMoE` with a defused per-expert linear MoE block. |
| `glm4_moe_lite` | Replaces `Glm4MoeLiteMoE` with a defused per-expert linear MoE block.|
| `glm4v` | Replaces the fused text MLP with split `gate_proj`, `up_proj`, and `down_proj` layers. Also splits fused checkpoint `mlp.gate_up_proj.weight` into `mlp.gate_proj.weight` + `mlp.up_proj.weight`. |
| `mixtral` | Replaces `MixtralSparseMoeBlock` with `LinearMixtralSparseMoeBlock`. Also remaps legacy Mixtral checkpoint keys and splits fused expert `gate_up_proj` tensors into per-expert `gate_proj` and `up_proj`, plus per-expert `down_proj`. |
| `qwen2_moe` | Replaces `Qwen2MoeSparseMoeBlock` with a defused per-expert linear MoE block. |
| `qwen3_moe` | Replaces `Qwen3MoeSparseMoeBlock` with a defused per-expert linear MoE block. |
| `qwen3_next` | Replaces `Qwen3NextSparseMoeBlock` with a defused per-expert linear MoE block. |
| `qwen3_omni_moe` | Replaces both thinker and talker text sparse MoE blocks with defused per-expert linear blocks and applies small runtime compatibility patches for text `forward()` and `generate()`. |

### 🔄 `convert_model(model)` after load

| Pattern | Supported model types | Defused op performed ⚙️ |
| --- | --- | --- |
| Standard routed expert tensors 🧱 | `deepseek_v2`, `dots1`, `ernie4_5_moe`, `ernie4_5_vl_moe`, `exaone_moe`, `flex_olmo`, `glm4_moe_lite`, `glm4v_moe`, `hunyuan_v1_moe`, `jamba`, `laguna`, `lfm2_moe`, `minimax`, `minimax_m2`, `olmoe`, `qwen3_vl_moe`, `solar_open` | Splits fused expert tensors or registered expert buffers into numbered expert `nn.Linear` modules with per-expert `gate_proj`, `up_proj`, and `down_proj`. |
| Mixed sparse and shared experts | `deepseek_v3`, `glm_moe_dsa`, `qwen3_5_moe`, `qwen3_5_moe_text` | Runtime expert tensor defusion for routed experts while preserving the model's shared-expert path. |
| Transposed or packed expert tensors | `gpt_oss`, `phimoe` | Splits transposed fused expert `gate_up_proj` tensors into per-expert `gate_proj` + `up_proj`, preserves expert bias when present, and converts expert tensors into numbered expert `nn.Linear` modules. |
| Flattened expert layout | `dbrx` | Rebuilds the flattened DBRX expert FFN weights into numbered expert `gate_proj`, `up_proj`, and `down_proj` `nn.Linear` modules. |
| Batched expert-input execution | `llama4` | Runtime expert tensor defusion plus preservation of the llama4 batched expert-input execution contract. |
| Non-gated expert MLPs | `nemotron_h` | Converts routed expert tensors into numbered `up_proj` and `down_proj` `nn.Linear` modules for non-gated experts. |
| Parallel expert blocks | `granitemoe`, `granitemoehybrid`, `granitemoeshared`, `jetmoe` | Converts packed expert weight tensors into numbered expert `linear` modules while keeping grouped expert execution intact. |
| Routed experts with identity experts | `longcat_flash` | Defuses routed experts into numbered `gate_proj`, `up_proj`, and `down_proj` modules and preserves zero or identity experts. |
| Fused dense `gate_up_proj` MLPs | `dia`, `glm`, `glm4`, `glm_image`, `glm_ocr`, `phi3`, `phi4_multimodal`, `zamba2` | Splits fused dense `gate_up_proj` layers into `gate_proj` + `up_proj` and updates the block `forward()` to preserve the original MLP math. |

## 🔁 Workflow Summary

Use `replace_fused_blocks()` for model families that Defuser can patch before load:

```python
from defuser import replace_fused_blocks
from transformers import MixtralForCausalLM

replace_fused_blocks("mixtral")
model = MixtralForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    dtype="auto",
    device_map="auto",
)
```

Use `convert_model()` for already loaded models whose expert tensors still need runtime defusion:

```python
from defuser import convert_model

converted = convert_model(model)
print(converted)  # True when runtime defusion happened
```

`convert_model(model)` also preserves meta-device construction for supported meta-initialized models, so structural validation can run without materializing weights.

Use `filter` when only specific blocks should be defused 🎯:

```python
from defuser import convert_model

convert_model(
    model,
    filter=[
        r"+:^model\.layers\.0\.mlp\.experts$",
        r"-:^model\.layers\.0\.mlp\.experts\.shared_",
    ],
)
```

## 🧪 Real Qwen3.5 MoE Example

The example below is written for the `transformers==5.3.0` public API surface and uses the real Hugging Face model `Qwen/Qwen3.5-35B-A3B-Instruct`. Defuser supports `transformers>=5.3.0`.

### 🔬 Fused Weights Before And After

Before `convert_model(model)`:

```text
+--------------------------------------------------------+---------------------------------------------+
| State dict key                                         | Layout                                      |
+--------------------------------------------------------+---------------------------------------------+
| model.language_model.layers.0.mlp.experts.gate_up_proj | fused gate+up tensor for all experts        |
|                                                        | [num_experts, 2 * moe_intermediate, hidden] |
| model.language_model.layers.0.mlp.experts.down_proj    | fused per-expert down tensor                |
|                                                        | [num_experts, hidden, moe_intermediate]     |
+--------------------------------------------------------+---------------------------------------------+
```

After `convert_model(model)`:

```text
+-----------------------------------------------------------------+--------------------------------------+
| State dict key                                                  | Layout                               |
+-----------------------------------------------------------------+--------------------------------------+
| model.language_model.layers.0.mlp.experts.0.gate_proj.weight    | expert 0 gate projection             |
| model.language_model.layers.0.mlp.experts.0.up_proj.weight      | expert 0 up projection               |
| model.language_model.layers.0.mlp.experts.0.down_proj.weight    | expert 0 down projection             |
| ... repeated for experts 1..N-1                                 | numbered expert nn.Linear modules    |
+-----------------------------------------------------------------+--------------------------------------+
```

### 🧭 Sample 1: Inspect The Conversion In Place

```python
from defuser import convert_model
from transformers import Qwen3_5MoeForConditionalGeneration

model_id = "Qwen/Qwen3.5-35B-A3B-Instruct"

model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)

prefix = "model.language_model.layers.0.mlp.experts"

before = [name for name, _ in model.named_parameters() if name.startswith(prefix)]
print(before)
# [
#   "model.language_model.layers.0.mlp.experts.gate_up_proj",
#   "model.language_model.layers.0.mlp.experts.down_proj",
# ]

converted = convert_model(model)
assert converted is True

after = [name for name, _ in model.named_parameters() if name.startswith(prefix)]
print(after[:6])
# [
#   "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
#   "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
#   "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
#   "model.language_model.layers.0.mlp.experts.1.down_proj.weight",
#   "model.language_model.layers.0.mlp.experts.1.gate_proj.weight",
#   "model.language_model.layers.0.mlp.experts.1.up_proj.weight",
# ]
```

### 🚀 Sample 2: Convert And Keep Using The Model Normally

```python
import torch

from defuser import convert_model
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

model_id = "Qwen/Qwen3.5-35B-A3B-Instruct"

model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

convert_model(model)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Explain mixture-of-experts routing in one sentence."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=64)

generated_ids = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
]
text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]
print(text)
```

After conversion, the first routed expert in the first MoE layer is exposed as normal submodules:

```python
expert0 = model.model.language_model.layers[0].mlp.experts[0]
print(type(expert0.gate_proj).__name__)  # Linear
print(type(expert0.up_proj).__name__)    # Linear
print(type(expert0.down_proj).__name__)  # Linear
```
