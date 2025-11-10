#!/usr/bin/env python3
"""
Test script for Gemini translation functionality
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini_translation():
    """Test Gemini translation functionality"""
    
    # Get API key
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in .env file")
        return False
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini model initialized successfully")
        
        # Test Gujarati to English translation
        gujarati_text = "તમે કેવા છો?"  # "How are you?" in Gujarati
        
        translation_prompt = f"""
Translate the following Gujarati text to English. Provide only the English translation, nothing else.

Gujarati text: {gujarati_text}

English translation:
        """
        
        response = model.generate_content(translation_prompt)
        english_translation = response.text.strip()
        print(f"🔄 Gujarati to English: '{gujarati_text}' -> '{english_translation}'")
        
        # Test English to Gujarati translation
        english_text = "Hello! Welcome to Brookstone. How can I help you today?"
        
        translation_prompt = f"""
Translate the following English text to Gujarati. Keep the same tone, style, and LENGTH - make it brief and concise like the original. Provide only the Gujarati translation, nothing else.

English text: {english_text}

Gujarati translation (keep it brief and concise):
        """
        
        response = model.generate_content(translation_prompt)
        gujarati_translation = response.text.strip()
        print(f"🔄 English to Gujarati: '{english_text}' -> '{gujarati_translation}'")
        
        print("✅ Gemini translation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini translation: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini Translation Functionality...")
    test_gemini_translation()