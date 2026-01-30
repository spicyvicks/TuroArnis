import os
import re
from pathlib import Path

#script to clean up all python file comments
#makes them lowercase, concise, and removes space after #

def clean_comment(line):
    #check if line contains a comment
    if '#' not in line:
        return line
    
    #split line into code and comment parts
    parts = line.split('#', 1)
    if len(parts) != 2:
        return line
    
    code_part = parts[0]
    comment_part = parts[1]
    
    #skip if this is a shebang line
    if line.strip().startswith('#!/'):
        return line
    
    #clean the comment
    #remove leading/trailing whitespace
    comment_cleaned = comment_part.strip()
    
    #convert to lowercase
    comment_cleaned = comment_cleaned.lower()
    
    #remove unnecessary """" and *** decorations
    comment_cleaned = comment_cleaned.replace('"""', '')
    comment_cleaned = comment_cleaned.replace('***', '')
    comment_cleaned = comment_cleaned.replace('**', '')
    comment_cleaned = comment_cleaned.strip()
    
    #rebuild line with cleaned comment (no space after #)
    if comment_cleaned:
        return f"{code_part}#{comment_cleaned}\n"
    else:
        return code_part.rstrip() + '\n'

def process_file(file_path):
    print(f"processing: {file_path}")
    
    #read file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    #process each line
    new_lines = []
    in_multiline_string = False
    quote_char = None
    
    for line in lines:
        #check if we're in a multiline string
        stripped = line.strip()
        
        #handle multiline strings (don't modify comments inside strings)
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote_char = '"""' if '"""' in line else "'''"
            if in_multiline_string and quote_char in line[3:]:
                in_multiline_string = False
            else:
                in_multiline_string = not in_multiline_string
            new_lines.append(line)
            continue
        
        if in_multiline_string:
            new_lines.append(line)
            continue
        
        #clean the comment in this line
        cleaned_line = clean_comment(line)
        new_lines.append(cleaned_line)
    
    #write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def main():
    project_root = Path(__file__).parent
    
    #find all .py files
    py_files = list(project_root.rglob('*.py'))
    
    #exclude this script itself
    py_files = [f for f in py_files if f.name != 'cleanup_comments.py']
    
    print(f"found {len(py_files)} python files to process")
    
    for py_file in py_files:
        try:
            process_file(py_file)
        except Exception as e:
            print(f"error processing {py_file}: {e}")
    
    print("cleanup complete!")

if __name__ == '__main__':
    main()
