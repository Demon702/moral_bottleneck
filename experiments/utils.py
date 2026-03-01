import re

def extract_answer_content(text):
    """
    Extract content between <answer> tags from the given text.
    
    Args:
        text (str): The input text containing <answer> tags
        
    Returns:
        str: The content between <answer> tags, or None if not found
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Example usage:
if __name__ == "__main__":
    sample_text = """
    Some text before
    <answer>
    {
        "explanation": "This is an example",
        "moral acceptability score": "2.5"
    }
    </answer>
    Some text after
    """
    
    result = extract_answer_content(sample_text)
    print(result) 