import json
from pathlib import Path



def load_prompt(filepath: Path, section: str= None, **kwargs) -> str:
    """
    Load prompt content from a markdown file.
    Supports variable substitution using {variable_name} syntax.
    Can extract specific sections by header name.
    
    Args:
        filepath: Path to the markdown file
        section: Optional section header to extract (without ##)
        **kwargs: Variables to inject into the prompt
    
    Returns:
        Full content or specific section content with variables injected
    
    Examples:
        # Load entire file
        load_prompt(Path("prompt.md"), name="John")
        
        # Load specific section
        load_prompt(Path("prompt.md"), section="Analysis Instructions", name="John")
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract specific section if requested
    if section:
        content = _extract_section(content, section)
    
    # Inject variables
    if kwargs:
        content = content.format(**kwargs)
    
    return content


def _extract_section(content: str, section_name: str) -> str:
    """
    Extract a specific section from markdown content by header name.
    Supports ## headers at any level.
    
    Args:
        content: Full markdown content
        section_name: Header name to find (without ## prefix)
    
    Returns:
        Content of the section (excluding the header itself)
    
    Raises:
        ValueError: If section is not found
    """
    lines = content.split('\n')
    section_lines = []
    in_section = False
    section_level = None
    
    for line in lines:
        # Check if this is a header line
        if line.strip().startswith('#'):
            # Parse header level and title
            header_match = line.strip().lstrip('#')
            header_level = len(line.strip()) - len(header_match)
            header_title = header_match.strip()
            
            # Check if this is our target section
            if header_title.lower() == section_name.lower():
                in_section = True
                section_level = header_level
                continue  # Skip the header itself
            
            # If we're in a section and hit a same/higher level header, we're done
            elif in_section and header_level <= section_level:
                break
        
        # Collect lines if we're in the target section
        if in_section:
            section_lines.append(line)
    
    if not section_lines:
        raise ValueError(f"Section '{section_name}' not found in markdown file")
    
    return '\n'.join(section_lines).strip()