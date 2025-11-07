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

# ================================================
# CONVERSATION STATE
# ================================================
CONV_STATE = {}

def ensure_conversation_state(from_phone):
    """Ensure conversation state has all required fields"""
    if from_phone not in CONV_STATE:
        CONV_STATE[from_phone] = {"chat_history": [], "language": "english", "waiting_for": None}
    elif "waiting_for" not in CONV_STATE[from_phone]:
        CONV_STATE[from_phone]["waiting_for"] = None

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

def send_whatsapp_location(to_phone):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "location",
        "location": {
            "latitude": "23.0433468",
            "longitude": "72.4594457",
            "name": "Brookstone",
            "address": "Brookstone, Vaikunth Bungalows, Beside DPS Bopal Rd, next to A. Shridhar Oxygen Park, Bopal, Shilaj, Ahmedabad, Gujarat 380058"
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Location sent to {to_phone}")
        else:
            logging.error(f"❌ Failed to send location: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error sending location: {e}")

def send_whatsapp_document(to_phone, caption="Here is your Brookstone Brochure 📄"):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "document",
        "document": {"link": BROCHURE_URL, "caption": caption, "filename": "Brookstone_Brochure.pdf"}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            logging.info(f"✅ Document sent to {to_phone}")
        else:
            logging.error(f"❌ Failed to send document: {response.status_code} - {response.text}")
    except Exception as e:
        logging.error(f"❌ Error sending document: {e}")

def mark_message_as_read(message_id):
    url = f"https://graph.facebook.com/v23.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "status": "read", "message_id": message_id}
    try:
        requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e:
        logging.error(f"Error marking message as read: {e}")

# ================================================
# MESSAGE PROCESSING
# ================================================
def process_incoming_message(from_phone, message_text, message_id):
    ensure_conversation_state(from_phone)
    state = CONV_STATE[from_phone]
    guj = any("\u0A80" <= c <= "\u0AFF" for c in message_text)
    state["language"] = "gujarati" if guj else "english"
    state["chat_history"].append({"role": "user", "content": message_text})

    logging.info(f"📱 Processing message from {from_phone}: {message_text} [Language: {state['language']}]")

    # Check for follow-up responses
    message_lower = message_text.lower().strip()
    
    # Handle location confirmation
    if state.get("waiting_for") == "location_confirmation":
        if any(word in message_lower for word in ["yes", "yeah", "yep", "sure", "please", "ok", "okay", "send", "हाँ", "હા"]):
            state["waiting_for"] = None
            send_whatsapp_location(from_phone)
            send_whatsapp_text(from_phone, "📍 Here's our location! We're open from 10:30 AM to 7:00 PM. Looking forward to see you! 😊")
            return
        elif any(word in message_lower for word in ["no", "nope", "not now", "later", "નહીં", "ના"]):
            state["waiting_for"] = None
            send_whatsapp_text(from_phone, "No problem! Feel free to ask if you need anything else. You can contact our agents at 8238477697 or 9974812701 anytime! 😊")
            return
    
    # Handle site visit booking confirmation
    if state.get("waiting_for") == "site_visit_booking":
        if any(word in message_lower for word in ["yes", "yeah", "yep", "sure", "please", "book", "visit", "interested", "हाँ", "હા"]):
            state["waiting_for"] = None
            send_whatsapp_text(from_phone, "Perfect! Please contact *Mr. Nilesh at 7600612701* to book your site visit. He'll help you schedule a convenient time. Our site office is open from 10:30 AM to 7:00 PM. 📞")
            return
        elif any(word in message_lower for word in ["no", "not now", "later", "maybe", "નહીં", "ના"]):
            state["waiting_for"] = None
            send_whatsapp_text(from_phone, "No worries! Take your time. When you're ready to visit, just let me know or contact Mr. Nilesh at 7600612701. Is there anything else about Brookstone I can help you with? 😊")
            return
    
    # Handle brochure confirmation
    if state.get("waiting_for") == "brochure_confirmation":
        if any(word in message_lower for word in ["yes", "yeah", "yep", "sure", "please", "send", "brochure", "pdf", "हाँ", "હા"]):
            state["waiting_for"] = None
            send_whatsapp_document(from_phone)
            send_whatsapp_text(from_phone, "📄 Here's your Brookstone brochure! It has all the details about our luxury 3&4BHK flats. Any questions after going through it? 😊")
            return
        elif any(word in message_lower for word in ["no", "not now", "later", "નહીં", "ના"]):
            state["waiting_for"] = None
            send_whatsapp_text(from_phone, "Sure! Let me know if you'd like the brochure later or have any other questions about Brookstone. 😊")
            return

    if not retriever:
        send_whatsapp_text(from_phone, "Please contact our agents at 8238477697 or 9974812701 for more info.")
        return

    try:
        docs = retriever.invoke(message_text)
        logging.info(f"📚 Retrieved {len(docs)} relevant documents")

        context = "\n\n".join(
            [(d.page_content or "") + ("\n" + "\n".join(f"{k}: {v}" for k, v in (d.metadata or {}).items())) for d in docs]
        )

        system_prompt = f"""
You are a friendly real estate assistant for Brookstone project. Be conversational and natural like a helpful friend.

Answer **only** using the context below.
If something is not mentioned, say you don't have that information and suggest contacting the agent.
Ask follow-up questions and try to convince the user.
In follow-up questions, do not ask what is not present in knowledge base or pinecone. Ask only information what is present with us.
After follow-up questions by bot, if user says yes for the question you asked, provide the correct answer and continue the conversational and natural flow with user.
Be concise and direct - don't give overly detailed explanations, but include all relevant facts
For general BHK interest: "Brookstone has luxury 3&4BHK flats 🏠 What would you like to know - size, location, or amenities? (you have to say both 3&4BHK)"
Use 1-2 emojis maximum
End with short, natural follow-up
Always provide complete information when asked - don't cut off important details to make responses shorter

TIMING RESPONSES:
- For ANY timing related questions (office hours, site visiting times, when to visit, office timings, site timings): Always respond with "*10:30 AM to 7:00 PM*" and ask if they want the location
- Examples: "Our site office is open from *10:30 AM to 7:00 PM* every day. Would you like me to send you the *location*?"

ELEVATOR/LIFT RESPONSES:
- For structure/material questions: "KONE/SCHINDLER or equivalent"
- For ground floor lift questions: Only mention Block A and Block B lifts
- For "Are lifts available in all towers?": "Yes, each tower is equipped with premium elevators ensuring smooth mobility"
- Use the specific elevator_response from PROJECT DATA when available

SITE VISIT BOOKING:
- When users show interest in visiting (words like "visit", "see", "tour", "check out"): Ask if they'd like to book a site visit
- If they confirm interest: "Would you like me to help you book a site visit?"

BROCHURE SENDING:
- When users ask about details, sizes, floor plans, amenities: Ask if they'd like the brochure
- "Would you like me to send you our detailed brochure with all specifications?"

Examples:
- If user asks "what are the timings?" or "when can I visit?" or "office hours":
  Reply: "Our site office is open from *10:30 AM to 7:00 PM* every day. Would you like me to send you the *location*?"
- If user asks "Do you have 4BHK flats?":
  Reply: "Sure! Brookstone offers luxurious 3&4BHK flats. Would you like to know more about sizes, amenities, or availability?"
- If user shows interest in visiting: "Would you like me to help you book a site visit?"
- If user asks about pricing and it's not in context: "For latest pricing, please contact our agents directly."

Important: When you ask about location, brochure, or site visit booking - do NOT provide the answer immediately. Wait for user confirmation.

Carry out a friendly, conversational flow.
Do **NOT** invent or guess details.
Keep responses concise and WhatsApp-friendly (avoid markdown formatting).

---
Context:
{context}

User: {message_text}
Assistant:
        """.strip()

        response = llm.invoke(system_prompt).content.strip()
        logging.info(f"🧠 LLM Response: {response}")

        # --- Send primary text response ---
        send_whatsapp_text(from_phone, response)

        # --- Set conversation states based on bot's response ---
        response_lower = response.lower()
        
        # Check if bot is asking for location confirmation
        if "would you like me to send" in response_lower and "location" in response_lower:
            state["waiting_for"] = "location_confirmation"
            logging.info(f"🎯 Set state to location_confirmation for {from_phone}")
        
        # Check if bot is asking for site visit booking
        elif "would you like" in response_lower and ("book" in response_lower or "site visit" in response_lower):
            state["waiting_for"] = "site_visit_booking"
            logging.info(f"🎯 Set state to site_visit_booking for {from_phone}")
        
        # Check if bot is asking for brochure
        elif "would you like" in response_lower and ("brochure" in response_lower or "send you" in response_lower):
            state["waiting_for"] = "brochure_confirmation"
            logging.info(f"🎯 Set state to brochure_confirmation for {from_phone}")

        # Legacy intent detection (keep for backward compatibility)
        if re.search(r"\bsend.*location\b|\bhere.*location\b", response_lower) and state.get("waiting_for") != "location_confirmation":
            logging.info(f"📍 Legacy location trigger for {from_phone}")
            send_whatsapp_location(from_phone)

        elif re.search(r"\bhere.*brochure\b|\bsending.*brochure\b", response_lower) and state.get("waiting_for") != "brochure_confirmation":
            logging.info(f"📄 Legacy brochure trigger for {from_phone}")
            send_whatsapp_document(from_phone)

        state["chat_history"].append({"role": "assistant", "content": response})

    except Exception as e:
        logging.error(f"❌ Error in RAG processing: {e}")
        send_whatsapp_text(from_phone, "Sorry, I'm facing a technical issue. Please contact 8238477697 / 9974812701.")

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
    logging.info("Incoming webhook data")

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

                    mark_message_as_read(message_id)
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
        "pinecone_configured": bool(PINECONE_API_KEY)
    }), 200

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Brookstone WhatsApp RAG Bot is running!",
        "brochure_url": BROCHURE_URL,
        "endpoints": {"webhook": "/webhook", "health": "/health"}
    }), 200

# ================================================
# RUN APP
# ================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Starting Brookstone WhatsApp Bot on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
