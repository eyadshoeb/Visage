import os
import argparse
import pandas as pd
import time
from groq import Groq
from tqdm import tqdm

api_key = os.getenv("GROQ_API_KEY")

def explode_category_with_groq(client, text, category_name):
    prompt = f"""
    You are a security researcher simulating phishing attacks.
    The following email uses the psychological category: "{category_name}".
    Original: "{text}"
    
    Task: Write 5 distinct variations of this email. 
    Keep the same category ("{category_name}") but change the scenario.
    Make them sound professional but threatening.
    
    Format: Output ONLY the 5 variations separated by "|||". Do not add numbering.
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content.split("|||")
    except Exception as e:
        print(f"API Error: {e}")
        time.sleep(2)
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Phishing Data using Groq")
    parser.add_argument("--input", type=str, required=True, help="Path to input seed CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV")
    args = parser.parse_args()
    if not api_key:
        raise ValueError("Error: GROQ_API_KEY not found.")
    client = Groq(api_key=api_key)
    print(f"Loading seeds: {args.input}")
    try:
        df_seeds = pd.read_csv(args.input)
    except FileNotFoundError:
        print("Input file not found.")
        return
    new_rows = []
    print("Generating")
    
    for index, row in tqdm(df_seeds.iterrows(), total=len(df_seeds)):
        variations = explode_category_with_groq(client, row['text'], row['category'])
        
        for var in variations:
            clean_text = var.strip()
            if len(clean_text) > 10:
                new_rows.append({
                    'text': clean_text,
                    'label': 1,
                    'category': row['category']
                })

    df_augmented = pd.DataFrame(new_rows)
    df_augmented.to_csv(args.output, index=False)
    print(f"✅ Saved {len(df_augmented)} samples to {args.output}")

if __name__ == "__main__":
    main()