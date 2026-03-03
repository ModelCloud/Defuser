# Defuser
Model defuser helper for HF Transformers >= 5.0. In HF Transformers 5.x releases, many MoE modules became auto-stacked or auto-fused by new modeling code which has benefits but also downsides. 

* Goal is to provide naive module/layer forwarding code for all models supported by HF transformers where run-time weight and structure level optimizations such weight merging, stacking, fusing are reversed so the model is operating in a simple naive state. 
* There are cases, quantization libraries, where we need to run inference where module input/output needs to be individually captured and this pkg can help complete this task.  
