import os
import mimetypes
from pathlib import Path

def is_text_file(file_path):
    """Check if a file is likely to be a text file."""
    # Get the mimetype
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # If mimetype is None, check file extension
    if mime_type is None:
        # List of common text file extensions
        text_extensions = {
            '.py', '.txt', '.md', '.json', '.yaml', '.yml', '.cpp', '.h', '.hpp', 
            '.c', '.cc', '.js', '.html', '.css', '.sh', '.bash', '.zsh', '.fish',
            '.rst', '.ini', '.cfg', '.conf', '.xml', '.csv', '.tsv', '.sql'
        }
        return Path(file_path).suffix.lower() in text_extensions
    
    # Check if mimetype starts with 'text/'
    return mime_type.startswith('text/')

def compose_codebase(root_dir, output_file):
    """Compose all text files in the codebase into a single file."""
    root_path = Path(root_dir)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk through all directories and files
        for root, dirs, files in os.walk(root_path):
            for file in files:
                file_path = Path(root) / file
                
                # Skip if not a text file
                if not is_text_file(file_path):
                    continue
                
                try:
                    # Get relative path from root directory
                    rel_path = file_path.relative_to(root_path)
                    
                    # Write header
                    outfile.write(f"\n{'='*80}\n")
                    outfile.write(f"CONTENTS OF: {rel_path}\n")
                    outfile.write(f"{'='*80}\n\n")
                    
                    # Read and write file contents
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        outfile.write("\n")
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Define paths
    root_dir = "/host/rl_for_curobo/curobo"
    output_file = "composed_codebase.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Compose the codebase
    compose_codebase(root_dir, output_file)
    print(f"Codebase has been composed into {output_file}") 