"""Safety guards. These are NOT classification: they only stop the bot from asking for or
revealing private data, and from claiming live status."""
import re

CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
CRED_REQUEST_RE = re.compile(
    r"\b(tell|give|send|share|show|reveal|read|what('s| is)( the| my)?)\b.{0,30}\b(password|passcode|otp|cvv|pin)\b", re.I)
RESET_RE = re.compile(r"\b(reset|change|forgot|forgotten|recover|update|set up|create)\b", re.I)
LIVE_RE = re.compile(
    r"\b(live|real[- ]time|right now)\b.{0,25}\bstatus\b"
    r"|\bstatus\b.{0,25}\b(order|claim|policy|shipment|application)\b\W{0,3}(number|no\.?|id|#)?\W{0,3}[A-Za-z]{0,3}\d{3,}", re.I)
LIVE_CLAIM_RE = re.compile(
    r"\b(is currently|has been (shipped|delivered|approved|processed)|will arrive (on|by)|your (order|claim) (is|has))\b", re.I)

CREDENTIAL_REFUSAL = ("For your security I can't share or ask for passwords, one-time codes or card details. "
                      "Please use the official reset or verification process on our website, or contact our support team.")
LIVE_REPLY = ("I can't see live order or claim records from here. Please check your account page "
              "or contact our support team for the current status.")
SMALLTALK_REPLY = "Hello! How can I help you with your insurance questions today?"
OUT_OF_SCOPE_REPLY = ("Sorry, I can only help with insurance customer-support questions, such as policies, "
                      "claims, payments and account help.")
FALLBACK_MESSAGE = ("I could not find a reliable answer to that in our help content. Please contact our "
                    "support team or visit our website for help with this question.")


def mask_cards(text: str) -> str:
    return CARD_RE.sub("[CARD NUMBER]", text)
