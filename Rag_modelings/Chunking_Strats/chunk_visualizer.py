import os
import json
import re
from collections import defaultdict

def visualize_chunks_html(document, chunks, output_path="visualization.html", title="Chunk Visualization"):
    """
    Create an HTML visualization of chunks in a document, mapping them correctly to the JSON chunk list.
    """
    # Sort chunks by occurrence in document
    chunk_positions = []
    for i, chunk in enumerate(chunks):
        start_idx = document.find(chunk)
        if start_idx != -1:
            chunk_positions.append((start_idx, start_idx + len(chunk), i))
    
    # Sort by start position
    chunk_positions.sort()
    
    # Build the highlighted document
    output_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.5; }}
            .chunk {{ display: inline; padding: 2px; border-radius: 3px; }}
            .overlap {{ background-color: yellow; }}
    """.format(title)
    
    # Define colors for chunk groups
    colors = ["#FFCCCC", "#CCFFCC", "#CCCCFF", "#FFFFCC", "#FFCCFF", "#CCFFFF", "#FFDDBB", "#DDBBFF", "#BBFFDD", "#DDFFBB"]
    for i in range(10):
        output_html += ".chunk{} {{ background-color: {}; }}\n".format(i, colors[i % len(colors)])
    
    output_html += """
        </style>
    </head>
    <body>
        <h1>{}</h1>
        <pre>
    """.format(title)
    
    # Insert chunk highlights
    last_idx = 0
    for start, end, chunk_id in chunk_positions:
        output_html += document[last_idx:start]
        output_html += '<span class="chunk chunk{}">{}</span>'.format(chunk_id % 10, document[start:end])
        last_idx = end
    
    output_html += document[last_idx:]
    output_html += """
        </pre>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_html)
    
    print(f"Visualization saved to {output_path}")
    return output_path
