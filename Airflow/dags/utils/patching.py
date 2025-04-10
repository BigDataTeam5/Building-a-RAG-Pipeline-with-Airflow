import huggingface_hub
import functools

# Fix for "TypeError: hf_hub_download() got an unexpected keyword argument 'url'"
def patched_cached_download(*args, **kwargs):
    # Remove 'url' parameter if present as it's not supported in hf_hub_download
    if 'url' in kwargs:
        del kwargs['url']
    return huggingface_hub.hf_hub_download(*args, **kwargs)

# Add the missing function that sentence-transformers needs with proper handling
if not hasattr(huggingface_hub, 'cached_download'):
    huggingface_hub.cached_download = patched_cached_download
    print("Patched huggingface_hub.cached_download with url parameter handling")