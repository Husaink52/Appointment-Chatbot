import streamlit as st
import dateparser
import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import timedelta


# ----------------------- CONFIG -----------------------
st.set_page_config(page_title="TailorTalk Chatbot", page_icon="🧵")

SCOPES = ['https://www.googleapis.com/auth/calendar']
SERVICE_ACCOUNT_FILE = 'credentials.json'  # <- Your key file

# --------------------- CALENDAR SETUP ---------------------
def create_event(summary, start_time):
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build('calendar', 'v3', credentials=credentials)

    end_time = start_time + timedelta(hours=1)

    event = {
        'summary': summary,
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'Asia/Kolkata',
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'Asia/Kolkata',
        },
    }

    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return created_event.get('htmlLink')

# -------------------- LLM SETUP --------------------------
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

chat_model = load_model()

# -------------------- INTENT CLASSIFIER ------------------
def classify_intent(message: str) -> str:
    msg = message.lower()
    if any(word in msg for word in ["book", "schedule", "appointment"]):
        return "book"
    elif any(word in msg for word in ["cancel"]):
        return "cancel"
    elif any(word in msg for word in ["reschedule", "change"]):
        return "reschedule"
    elif any(word in msg for word in ["open", "time", "available", "hours"]):
        return "inquiry"
    return "chat"

# -------------------- STREAMLIT UI -----------------------
st.title("🤖 TailorTalk Chatbot")
st.markdown("Chat with me to ask questions or book appointments.")

# Maintain conversation history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if user_input:
    intent = classify_intent(user_input)

    if intent == "book":
        dt = dateparser.parse(user_input)
        if dt:
            try:
                link = create_event("Tailor Appointment", dt)
                reply = f"✅ Your appointment is booked for {dt.strftime('%A %d %B %Y at %I:%M %p')}.\n📅 [View in Calendar]({link})"
            except Exception as e:
                reply = f"❌ Failed to book: {str(e)}"
        else:
            reply = "❌ I couldn't understand the time/date. Please rephrase."
    
    elif intent == "cancel":
        reply = "Your appointment has been canceled (not really, just a placeholder)."
    
    elif intent == "reschedule":
        reply = "Sure, when would you like to reschedule to?"
    
    elif intent == "inquiry":
        reply = "We are open Monday to Saturday, 10 AM to 6 PM."
    
    else:
        reply = chat_model(user_input)[0]['generated_text']

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", reply))

# Display chat history
for sender, msg in st.session_state.history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(msg)
