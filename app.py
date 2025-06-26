import streamlit as st
import dateparser
import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import timedelta

# ------------------ Google Calendar Setup ------------------
import json
from google.oauth2 import service_account

service_account_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/calendar"]
)


def create_event(summary, start_time):
    end_time = start_time + timedelta(hours=1)
    event = {
        'summary': summary,
        'start': {'dateTime': start_time.isoformat(), 'timeZone': 'Asia/Kolkata'},
        'end': {'dateTime': end_time.isoformat(), 'timeZone': 'Asia/Kolkata'},
    }
    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return created_event.get('htmlLink')


# ------------------ Load HuggingFace Model ------------------

@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

chat_model = load_model()


# ------------------ Intent Classifier ------------------

def classify_intent(msg: str) -> str:
    msg = msg.lower()
    if any(x in msg for x in ["book", "schedule", "appointment"]):
        return "book"
    elif any(x in msg for x in ["cancel"]):
        return "cancel"
    elif any(x in msg for x in ["reschedule", "change"]):
        return "reschedule"
    elif any(x in msg for x in ["open", "available", "hours"]):
        return "inquiry"
    return "chat"


# ------------------ Streamlit Chat UI ------------------

st.set_page_config(page_title="TailorTalk Chatbot")
st.title("👔 TailorTalk Chatbot")
st.markdown("Ask questions or book your appointment!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")

if user_input:
    intent = classify_intent(user_input)

    if intent == "book":
        dt = dateparser.parse(user_input)
        if dt:
            try:
                event_link = create_event("Tailor Appointment", dt)
                reply = f"✅ Appointment booked for {dt.strftime('%A %d %B %Y at %I:%M %p')}.\n📅 [View in Calendar]({event_link})"
            except Exception as e:
                reply = f"❌ Booking failed: {e}"
        else:
            reply = "❌ I couldn't understand the date/time. Please try again."

    elif intent == "cancel":
        reply = "🗑️ Appointment canceled. (Placeholder)"
    elif intent == "reschedule":
        reply = "🔁 Sure. Please tell me the new date and time."
    elif intent == "inquiry":
        reply = "⏰ We're open Monday to Saturday, 10 AM to 6 PM."
    else:
        reply = chat_model(user_input)[0]['generated_text']

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", reply))

# Chat history display
for sender, msg in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(msg)
