import re

def remove_comments(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith('#'):
            continue
        line = re.sub(r'\s*#[^"\']*$', '', line)
        cleaned_lines.append(line)
    
    cleaned_content = ''.join(cleaned_lines)
    cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"Removed comments from {filepath}")

remove_comments('main_app.py')
remove_comments('main_image.py')
print("Done!")
