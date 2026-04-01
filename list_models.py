"""Run this to see which Gemini models your API key can access."""
import sys
from google import genai

key = input("Paste your Gemini API key: ").strip()
client = genai.Client(api_key=key)

print("\nAvailable models that support generateContent:\n")
for m in client.models.list():
    actions = getattr(m, "supported_actions", []) or []
    if "generateContent" in actions:
        print(f"  {m.name}")
