"""
sanity_check.py — Gate 0 Pass Criteria

Verifies that:
1. Environment is set up correctly
2. Groq API key is loaded from .env
3. A simple hello-world call to Groq returns text
"""

import os
import sys

# Fix Windows console encoding for emoji/unicode
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv

load_dotenv()

def main():
    # --- Check API key ---
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("GROQ_API_KEY not set in .env file!")
        print("   Go to https://console.groq.com to get your key,")
        print("   then update .env with: GROQ_API_KEY=gsk_...")
        return

    print(f"[OK] GROQ_API_KEY loaded (starts with {api_key[:8]}...)")

    # --- Test Groq API call ---
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0,
        api_key=api_key,
    )

    print("\n[...] Calling Groq API with a hello-world prompt...")
    try:
        response = llm.invoke("Say hello and confirm you are working. Keep it to one sentence.")
        print(f"\n[OK] Groq API Response:\n   {response.content}")
        print("\n>>> Gate 0 PASSED -- Environment & API are working!")
    except Exception as e:
        print(f"\n[ERROR] Groq API call failed: {e}")
        print("\nTrying alternative model 'openai/gpt-oss-20b'...")
        try:
            llm2 = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0,
                api_key=api_key,
            )
            response = llm2.invoke("Say hello and confirm you are working. Keep it to one sentence.")
            print(f"\n[OK] Groq API Response (gpt-oss-20b):\n   {response.content}")
            print("\n>>> Gate 0 PASSED -- Environment & API are working!")
        except Exception as e2:
            print(f"\n[ERROR] Fallback model also failed: {e2}")
            print("\nPlease check available models at https://console.groq.com/docs/models")


if __name__ == "__main__":
    main()
