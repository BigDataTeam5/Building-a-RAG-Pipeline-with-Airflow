import huggingface_hub

# Add the missing function that sentence-transformers needs
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    print("Patched huggingface_hub.cached_download")