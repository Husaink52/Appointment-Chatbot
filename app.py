import streamlit as st
import datetime
import dateparser
from dateutil import parser as du_parser
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import timedelta
import json

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="TailorTalk Chatbot", page_icon="🤖")
st.title(" TailorTalk Chatbot")
st.markdown("Ask questions or book your tailoring appointment!")

# ------------------ Google Calendar Setup ------------------
SCOPES = ['https://www.googleapis.com/auth/calendar']

service_account_info = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=SCOPES
)
service = build('calendar', 'v3', credentials=credentials)

def create_event(summary, start_time):
    try:
        end_time = start_time + timedelta(hours=1)
        event = {
            'summary': summary,
            'start': {'dateTime': start_time.isoformat(), 'timeZone': 'Asia/Kolkata'},
            'end': {'dateTime': end_time.isoformat(), 'timeZone': 'Asia/Kolkata'},
        }
        created_event = service.events().insert(calendarId='husainkhalapurwala52@gmail.com', body=event).execute()
        print("Event created:", created_event)
        return created_event.get('htmlLink')
    except Exception as e:
        print("Error creating event:", e)
        return None


# ------------------ LLM Setup ------------------
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
    if any(x in msg for x in ["book", "schedule", "appointment","meeting"]):
        return "book"
    elif any(x in msg for x in ["cancel"]):
        return "cancel"
    elif any(x in msg for x in ["reschedule", "change"]):
        return "reschedule"
    elif any(x in msg for x in ["open", "available", "hours", "time"]):
        return "inquiry"
    return "chat"

# ------------------ Session State ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="input")

if user_input:
    intent = classify_intent(user_input)

    if intent == "book":
        dt = None
        try:
            # Try dateparser first
            dt = dateparser.parse(
                user_input,
                settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.datetime.now()}
            )
            # Fallback to dateutil if needed
            if not dt:
                dt = du_parser.parse(user_input, fuzzy=True)
        except Exception:
            dt = None

        if dt:
            try:
                link = create_event("Tailor Appointment", dt)
                reply = (
                    f" Appointment booked for {dt.strftime('%A %d %B %Y at %I:%M %p')}.\n"
                    f" [View in Calendar]({link})"
                )
            except Exception as e:
                reply = f" Booking failed: {e}"
        else:
            reply = (
                " Sorry, I couldn't extract a valid date/time.\n"
                "Try using formats like:\n"
                "• 28 June at 3pm\n"
                "• tomorrow at 4pm\n"
                "• next Monday 10:00"
            )

    elif intent == "cancel":
        reply = " Your appointment has been canceled. (This is a placeholder.)"

    elif intent == "reschedule":
        reply = " Sure! Please tell me the new date and time you'd like to reschedule to."

    elif intent == "inquiry":
        reply = " We're open Monday to Saturday, 10 AM to 6 PM."

    else:
        reply = chat_model(user_input)[0]['generated_text']

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", reply))

# ------------------ Chat History ------------------
for sender, msg in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(msg)
