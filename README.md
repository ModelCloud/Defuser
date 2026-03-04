
<div align=center>
<img width="50%" alt="image" src="https://github.com/user-attachments/assets/f801617b-8959-474a-a565-6b8897e2fcbf" />
<h1 align="center">Defuser</h1>
</div>

<p align="center">
    <a href="https://github.com/ModelCloud/Defuser/releases" style="text-decoration:none;"><img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/Defuser.svg"></a>
    <a href="https://pypi.org/project/Defuser/" style="text-decoration:none;"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/Defuser"></a>
    <a href="https://pepy.tech/projects/Defuser" style="text-decoration:none;"><img src="https://static.pepy.tech/badge/Defuser" alt="PyPI Downloads"></a>
    <a href="https://github.com/ModelCloud/Defuser/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/Defuser"></a>
    <a href="https://huggingface.co/modelcloud/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ModelCloud-%23ff8811.svg"></a>
</p>
Model defuser helper for HF Transformers >= 5.0. In HF Transformers 5.x releases, many MoE modules became auto-stacked or auto-fused by new modeling code which has benefits but also downsides. 

* Goal is to provide naive module/layer forwarding code for all models supported by HF transformers where run-time weight and structure level optimizations such weight merging, stacking, fusing are reversed so the model is operating in a simple naive state. 
* There are cases, quantization libraries, where we need to run inference where module input/output needs to be individually captured and this pkg can help complete this task.  
