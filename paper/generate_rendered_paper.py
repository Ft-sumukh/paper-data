import os
import sys
import re

def markdown_to_html(md_path="paper/manuscript.md", html_path="paper/manuscript_rendered.html"):
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process math to preserve KaTeX delimiters
    # Replace display math $$ ... $$ with placeholder
    display_math = []
    def save_display_math(match):
        idx = len(display_math)
        display_math.append(match.group(1))
        return f"<!--DISPLAY_MATH_{idx}-->"

    md_text = re.sub(r'\$\$(.*?)\$\$', save_display_math, md_text, flags=re.DOTALL)

    # Replace inline math $ ... $ with placeholder
    inline_math = []
    def save_inline_math(match):
        idx = len(inline_math)
        inline_math.append(match.group(1))
        return f"<!--INLINE_MATH_{idx}-->"

    md_text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', save_inline_math, md_text)

    # Process tables
    def format_table(block):
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return block
        headers = [c.strip() for c in lines[0].strip("|").split("|")]
        # separator line is lines[1]
        html = ['<div class="table-container"><table>']
        html.append('<thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead>')
        html.append('<tbody>')
        for line in lines[2:]:
            cols = [c.strip() for c in line.strip("|").split("|")]
            html.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cols) + '</tr>')
        html.append('</tbody></table></div>')
        return "\n".join(html)

    # Convert markdown tables
    table_pattern = re.compile(r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n?)+)')
    md_text = table_pattern.sub(lambda m: format_table(m.group(1)), md_text)

    # Convert images
    def format_image(match):
        alt = match.group(1)
        src = match.group(2)
        # Ensure path is relative to html location or absolute
        return f'<figure class="paper-figure"><img src="{src}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'

    md_text = re.sub(r'!\[(.*?)\]\((.*?)\)', format_image, md_text)

    # Convert headers
    md_text = re.sub(r'^# (.*?)$', r'<h1 class="paper-title">\1</h1>', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^## (.*?)$', r'<h2 class="section-title">\1</h2>', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^### (.*?)$', r'<h3 class="subsection-title">\1</h3>', md_text, flags=re.MULTILINE)
    md_text = re.sub(r'^#### (.*?)$', r'<h4 class="subsubsection-title">\1</h4>', md_text, flags=re.MULTILINE)

    # Convert bold and italic
    md_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', md_text)
    md_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', md_text)

    # Convert blockquotes
    md_text = re.sub(r'^> (.*?)$', r'<blockquote class="paper-quote">\1</blockquote>', md_text, flags=re.MULTILINE)

    # Convert horizontal rules
    md_text = re.sub(r'^---$', r'<hr class="paper-divider">', md_text, flags=re.MULTILINE)

    # Convert lists
    lines = md_text.split("\n")
    in_list = False
    new_lines = []
    for line in lines:
        if line.strip().startswith("- "):
            if not in_list:
                new_lines.append('<ul class="paper-list">')
                in_list = True
            item = line.strip()[2:]
            new_lines.append(f'<li>{item}</li>')
        elif re.match(r'^\d+\.\s', line.strip()):
            if not in_list:
                new_lines.append('<ol class="paper-list">')
                in_list = True
            item = re.sub(r'^\d+\.\s', '', line.strip())
            new_lines.append(f'<li>{item}</li>')
        else:
            if in_list:
                new_lines.append('</ul>')
                in_list = False
            new_lines.append(line)
    if in_list:
        new_lines.append('</ul>')
    md_text = "\n".join(new_lines)

    # Paragraphs (lines separated by double newlines)
    paragraphs = md_text.split("\n\n")
    processed_p = []
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        if p_str.startswith("<h") or p_str.startswith("<div") or p_str.startswith("<figure") or p_str.startswith("<hr") or p_str.startswith("<ul") or p_str.startswith("<ol") or p_str.startswith("<blockquote"):
            processed_p.append(p_str)
        else:
            processed_p.append(f'<p>{p_str}</p>')
    body_html = "\n\n".join(processed_p)

    # Restore math
    for idx, math in enumerate(display_math):
        # Clean math for katex
        clean_math = math.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body_html = body_html.replace(f"<!--DISPLAY_MATH_{idx}-->", f'<div class="katex-display">\\[{clean_math}\\]</div>')

    for idx, math in enumerate(inline_math):
        clean_math = math.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body_html = body_html.replace(f"<!--INLINE_MATH_{idx}-->", f'<span class="katex-inline">\\({clean_math}\\)</span>')

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Market Risk & Portfolio Optimization Under Different Market Regimes</title>
    
    <!-- KaTeX CSS and JS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '\\\\[', right: '\\\\]', display: true}},
                {{left: '$', right: '$', display: false}},
                {{left: '\\\\(', right: '\\\\)', display: false}}
            ]
        }});"></script>

    <style>
        :root {{
            --primary: #1a365d;
            --secondary: #2b6cb0;
            --text: #2d3748;
            --bg: #ffffff;
            --border: #e2e8f0;
            --accent: #c53030;
            --code-bg: #f7fafc;
        }}

        body {{
            font-family: 'Times New Roman', Times, 'Georgia', serif;
            color: var(--text);
            background-color: #f4f6f9;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }}

        .paper-container {{
            max-width: 960px;
            margin: 0 auto;
            background: var(--bg);
            padding: 50px 70px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border-radius: 4px;
        }}

        .paper-title {{
            font-size: 26pt;
            font-weight: bold;
            text-align: center;
            color: #111;
            margin-bottom: 12px;
            line-height: 1.25;
            font-family: 'Times New Roman', serif;
        }}

        .paper-meta {{
            text-align: center;
            font-size: 11pt;
            color: #4a5568;
            margin-bottom: 30px;
            border-bottom: 1.5px solid #2d3748;
            padding-bottom: 15px;
        }}

        .section-title {{
            font-size: 16pt;
            font-weight: bold;
            color: var(--primary);
            border-bottom: 1px solid #cbd5e0;
            padding-bottom: 4px;
            margin-top: 36px;
            margin-bottom: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .subsection-title {{
            font-size: 13pt;
            font-weight: bold;
            color: var(--secondary);
            margin-top: 24px;
            margin-bottom: 10px;
        }}

        .subsubsection-title {{
            font-size: 11pt;
            font-weight: bold;
            color: #4a5568;
            margin-top: 18px;
            margin-bottom: 6px;
        }}

        p {{
            text-align: justify;
            text-justify: inter-word;
            font-size: 11pt;
            margin-bottom: 14px;
        }}

        .paper-quote {{
            border-left: 4px solid var(--secondary);
            background: #f7fafc;
            padding: 12px 20px;
            margin: 18px 0;
            font-style: italic;
            color: #4a5568;
        }}

        /* Tables */
        .table-container {{
            overflow-x: auto;
            margin: 24px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 9.5pt;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-bottom: 12px;
        }}

        th {{
            background: #2b6cb0;
            color: white;
            padding: 8px 10px;
            text-align: left;
            font-weight: 600;
            border-top: 2px solid #1a365d;
            border-bottom: 2px solid #1a365d;
        }}

        td {{
            padding: 6px 10px;
            border-bottom: 1px solid #e2e8f0;
        }}

        tr:nth-child(even) td {{
            background: #f8fafc;
        }}

        tr:hover td {{
            background: #edf2f7;
        }}

        /* Figures */
        .paper-figure {{
            margin: 28px 0;
            text-align: center;
        }}

        .paper-figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}

        figcaption {{
            font-size: 9.5pt;
            font-style: italic;
            color: #4a5568;
            margin-top: 8px;
        }}

        .paper-list {{
            font-size: 10.5pt;
            margin-left: 20px;
            margin-bottom: 14px;
        }}

        .paper-list li {{
            margin-bottom: 6px;
        }}

        .paper-divider {{
            border: 0;
            border-top: 1px dashed #cbd5e0;
            margin: 30px 0;
        }}

        .katex-display {{
            margin: 16px 0 !important;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 6px 0;
        }}

        /* Print formatting */
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .paper-container {{
                box-shadow: none;
                padding: 0;
                max-width: 100%;
            }}
            .section-title {{
                page-break-after: avoid;
            }}
            table, .paper-figure {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="paper-container">
        {body_html}
    </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Successfully rendered paper to: {html_path} ({len(full_html)} bytes)")

if __name__ == "__main__":
    markdown_to_html()
