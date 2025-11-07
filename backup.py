import os
import re
import logging
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================================================
# ENVIRONMENT VARIABLES
# ================================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "brookstone_verify_token_2024")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
BROCHURE_URL = os.getenv("BROCHURE_URL", "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/BROOKSTONE.pdf")

# >>> Added for WorkVEU <<<
WORKVEU_WEBHOOK_URL = os.getenv("WORKVEU_WEBHOOK_URL")
WORKVEU_API_KEY = os.getenv("WORKVEU_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    logging.error("❌ Missing API keys!")

# ================================================
# PINECONE SETUP
# ================================================
INDEX_NAME = "brookstone-faq-json"

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 100})
    logging.info("✅ Pinecone vectorstore loaded successfully")
except Exception as e:
    logging.error(f"❌ Error loading Pinecone: {e}")
    retriever = None

# ================================================
# LLM SETUP
# ================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
translator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ================================================
# TRANSLATION FUNCTIONS
# ================================================
def translate_gujarati_to_english(text):
    try:
        translation_prompt = f"""
Translate the following Gujarati text to English. Provide only the English translation, nothing else.

Gujarati text: {text}

English translation:
        """
        response = translator_llm.invoke(translation_prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"❌ Error translating Gujarati to English: {e}")
        return text

def translate_english_to_gujarati(text):
    try:
        translation_prompt = f"""
Translate the following English text to Gujarati. Keep the same tone and style. Provide only the Gujarati translation, nothing else.

English text: {text}

Gujarati translation:
        """
        response = translator_llm.invoke(translation_prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"❌ Error translating English to Gujarati: {e}")
        return text

# ================================================
# CONVERSATION STATE & CONTEXT
# ================================================
CONV_STATE = {}

def ensure_conversation_state(from_phone):
    if from_phone not in CONV_STATE:
        CONV_STATE[from_phone] = {
            "chat_history": [], 
            "language": "english", 
            "waiting_for": None,
            "last_context_topics": [],
            "user_interests": []
        }
    else:
        if "waiting_for" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["waiting_for"] = None
        if "last_context_topics" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["last_context_topics"] = []
        if "user_interests" not in CONV_STATE[from_phone]:
            CONV_STATE[from_phone]["user_interests"] = []

def analyze_user_interests(message_text, state):
    message_lower = message_text.lower()
    interests = []
    interest_keywords = {
        "pricing": ["price", "cost", "budget", "expensive", "cheap", "affordable", "rate"],
        "size": ["size", "area", "bhk", "bedroom", "space", "sqft", "square"],
        "amenities": ["amenities", "facilities", "gym", "pool", "parking", "security"],
        "location": ["location", "address", "nearby", "connectivity", "metro", "airport"],
        "availability": ["available", "ready", "possession", "when", "booking"],
        "visit": ["visit", "see", "tour", "show", "check", "viewing"]
    }
    for category, keywords in interest_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            interests.append(category)
    state["user_interests"].extend(interests)
    state["user_interests"] = list(set(state["user_interests"][-5:]))
    return interests

# ================================================
# WHATSAPP FUNCTIONS
# ================================================
def send_whatsapp_text(to_phone, message):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": message}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Message sent to {to_phone}")
        else:
            logging.error(f"❌ Failed to send message: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error sending message: {e}")

# >>> Added for WorkVEU <<<
def push_to_workveu(name, wa_id, message_text, direction="inbound"):
    """Push chat messages to WorkVEU CRM"""
    if not WORKVEU_WEBHOOK_URL or not WORKVEU_API_KEY:
        logging.warning("⚠️ WorkVEU integration skipped: Missing configuration.")
        return

    payload = {
        "api_key": WORKVEU_API_KEY,
        "contacts": [
            {
                "profile": {"name": name or "Unknown"},
                "wa_id": wa_id,
                "remarks": f"[{direction.upper()}] {message_text}"
            }
        ]
    }

    try:
        headers = {"Content-Type": "application/json"}
        res = requests.post(WORKVEU_WEBHOOK_URL, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            logging.info(f"✅ WorkVEU message synced ({direction}) for {wa_id}")
        else:
            logging.error(f"❌ WorkVEU sync failed: {res.status_code} - {res.text}")
    except Exception as e:
        logging.error(f"❌ Error pushing to WorkVEU: {e}")

# ================================================
# MESSAGE PROCESSING
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    ensure_conversation_state(from_phone)
    state = CONV_STATE[from_phone]
    guj = any("\u0A80" <= c <= "\u0AFF" for c in message_text)
    state["language"] = "gujarati" if guj else "english"
    state["chat_history"].append({"role": "user", "content": message_text})

    # >>> Added for WorkVEU <<<
    push_to_workveu(name=None, wa_id=from_phone, message_text=message_text, direction="inbound")

    current_interests = analyze_user_interests(message_text, state)
    logging.info(f"📱 Message from {from_phone}: {message_text} [Language: {state['language']}]")

    # === (Existing logic continues unchanged up to sending response) ===

    try:
        search_query = message_text
        if state["language"] == "gujarati":
            search_query = translate_gujarati_to_english(message_text)

        docs = retriever.invoke(search_query)
        context = "\n\n".join([(d.page_content or "") for d in docs])

        system_prompt = f"""
You are a friendly Brookstone real estate assistant. Respond concisely and conversationally.

Context:
{context}

User Question: {search_query}
        """

        response = llm.invoke(system_prompt).content.strip()
        final_response = translate_english_to_gujarati(response) if state["language"] == "gujarati" else response

        # Send bot reply
        send_whatsapp_text(from_phone, final_response)

        # >>> Added for WorkVEU <<<
        push_to_workveu(name="Brookstone Bot", wa_id=from_phone, message_text=final_response, direction="outbound")

        state["chat_history"].append({"role": "assistant", "content": final_response})

    except Exception as e:
        logging.error(f"❌ Error in processing: {e}")
        send_whatsapp_text(from_phone, "Sorry, something went wrong. Please try again later.")

# ================================================
# WEBHOOK ROUTES
# ================================================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logging.info("✅ WEBHOOK VERIFIED")
        return challenge, 200
    else:
        return "Forbidden", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                for message in messages:
                    from_phone = message.get("from")
                    message_id = message.get("id")
                    msg_type = message.get("type")
                    text = ""
                    if msg_type == "text":
                        text = message.get("text", {}).get("body", "")
                    elif msg_type == "button":
                        text = message.get("button", {}).get("text", "")
                    elif msg_type == "interactive":
                        interactive = message.get("interactive", {})
                        if "button_reply" in interactive:
                            text = interactive["button_reply"].get("title", "")
                        elif "list_reply" in interactive:
                            text = interactive["list_reply"].get("title", "")
                    if not text:
                        continue
                    process_incoming_message(from_phone, text, message_id)
    except Exception as e:
        logging.exception("❌ Error processing webhook")
    return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "whatsapp_configured": bool(WHATSAPP_TOKEN and WHATSAPP_PHONE_NUMBER_ID),
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "workveu_configured": bool(WORKVEU_API_KEY)
    }), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Brookstone WhatsApp Bot is running!",
        "endpoints": {"webhook": "/webhook", "health": "/health"}
    }), 200

# ================================================
# RUN APP
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting Brookstone WhatsApp Bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
