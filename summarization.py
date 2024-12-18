from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from tqdm import tqdm  # For progress bar

# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def generate_summary(text):
    """
    Generates a summary for the given input text.

    Args:
        text (str): The input text to summarize.

    Returns:
        str: The generated summary.
    """
    try:
        # Ensure the text is not empty
        if not text.strip():
            raise ValueError("The input text is empty.")

        # Tokenize input, limit to 512 tokens, and ensure text is appropriately truncated
        input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

        # Print progress bar for long texts
        print("Generating summary...")
        summary_ids = model.generate(
            input_ids, 
            max_length=200,  # Adjust length as necessary
            min_length=50,
            length_penalty=3.0,  # Prevents too short summaries
            num_beams=4,  # Use 4 beams for better quality (can be reduced for speed)
            early_stopping=True
        )

        # Decode the summary and return it
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Return the generated summary or a message if it's too short or irrelevant
        if len(summary.split()) < 10:
            return "Summary seems too short to be meaningful."
        return summary

    except ValueError as ve:
        return f"Error: {str(ve)}"
    except Exception as e:
        return f"Error during summarization: {str(e)}"


def split_text_for_long_documents(text, max_length=512):
    """
    Splits long text into smaller chunks to avoid model input size limits.

    Args:
        text (str): The input text to split.
        max_length (int): Maximum length for each chunk (default is 512).

    Returns:
        list: A list of text chunks.
    """
    # Tokenize the text into tokens
    tokens = tokenizer.encode(text, truncation=False)
    
    # Split tokens into chunks of max_length tokens
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    
    # Convert token chunks back into text
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]


def generate_summary_from_long_text(text):
    """
    Handles long text by splitting it into chunks and summarizing each chunk.
    
    Args:
        text (str): The input text to summarize.
        
    Returns:
        str: The final generated summary.
    """
    chunks = split_text_for_long_documents(text)
    summaries = []

    # Summarize each chunk
    for chunk in tqdm(chunks, desc="Summarizing chunks", unit="chunk"):
        summaries.append(generate_summary(chunk))

    # Combine all the summaries and provide a final summary
    full_summary = " ".join(summaries)
    return full_summary


# Example usage
text = """Enter your long text here or provide a PDF to text input."""
summary = generate_summary_from_long_text(text)  # Use this for long texts
print("Final Summary:", summary)
