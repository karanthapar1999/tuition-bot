"""
WhatsApp Webhook Server for Tuition Bot
========================================
Handles incoming messages from Twilio WhatsApp and routes them to the tuition bot.
"""

import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import requests
from io import BytesIO

# Import your bot
from tuition_bot import process_message, process_image

# ============================================
# CONFIGURATION
# ============================================

# TWILIO CREDENTIALS - Get these from Twilio Console
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "YOUR_ACCOUNT_SID_HERE")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "YOUR_AUTH_TOKEN_HERE")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")  # Sandbox number

# Webhook verification token (you set this)
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_secret_token_123")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Initialize Flask app
app = Flask(__name__)

# ============================================
# HELPER FUNCTIONS
# ============================================

def download_media(media_url: str) -> bytes:
    """Download media from Twilio's URL"""
    try:
        # Twilio requires authentication to download media
        response = requests.get(
            media_url,
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=30
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Error downloading media: {e}")
        return None

def send_whatsapp_message(to_number: str, message: str):
    """Send a text message via WhatsApp"""
    try:
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=to_number
        )
        print(f"Sent message: {message.sid}")
        return message.sid
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

def send_whatsapp_image(to_number: str, image_bytes: bytes, caption: str = ""):
    """Send an image via WhatsApp"""
    try:
        # Save image temporarily (Twilio needs a URL)
        # For production, upload to S3/Cloud Storage
        # For now, we'll send as base64 in the message (not ideal but works for testing)
        
        # Better approach: Use Twilio's media URL
        # For testing, just send the caption text with a note
        message_text = f"{caption}\n\n[Image solution generated - will send as separate message in production]"
        
        message = twilio_client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_text,
            to=to_number
        )
        print(f"Sent image message: {message.sid}")
        return message.sid
    except Exception as e:
        print(f"Error sending image: {e}")
        return None

# ============================================
# WEBHOOK ENDPOINTS
# ============================================

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Verify webhook for initial setup"""
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    
    if token == VERIFY_TOKEN:
        return challenge
    return "Invalid verification token", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    """Main webhook endpoint for incoming WhatsApp messages"""
    try:
        # Get incoming message data
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        media_url = request.values.get('MediaUrl0', '')  # First media attachment
        media_content_type = request.values.get('MediaContentType0', '')
        
        # Extract just the phone number (remove 'whatsapp:' prefix)
        phone = from_number.replace('whatsapp:', '')
        
        print(f"\n{'='*60}")
        print(f"📱 Incoming from: {phone}")
        print(f"📝 Message: {incoming_msg}")
        print(f"🖼️  Media: {media_url}")
        print(f"{'='*60}\n")
        
        # Initialize response
        resp = MessagingResponse()
        
        # Handle image messages
        if media_url and 'image' in media_content_type:
            print("Processing image message...")
            
            # Download the image
            image_bytes = download_media(media_url)
            
            if image_bytes:
                # Process image with bot
                caption = incoming_msg if incoming_msg else ""
                response_dict = process_image(phone, image_bytes, caption)
                
                # Send response
                if 'image_bytes' in response_dict:
                    # Bot generated an image solution
                    send_whatsapp_message(phone, response_dict['text'])
                    # TODO: Upload image to cloud storage and send URL
                    # For now, text response includes the solution
                else:
                    # Text-only response
                    resp.message(response_dict['text'])
            else:
                resp.message("I couldn't download that image. Please try sending it again.")
        
        # Handle text messages
        elif incoming_msg:
            print("Processing text message...")
            
            # Process text with bot
            response_dict = process_message(phone, incoming_msg)
            
            # Send response
            if 'image_bytes' in response_dict:
                # Bot generated an image solution
                send_whatsapp_message(phone, response_dict['text'])
                # TODO: Upload image to cloud storage and send URL
            else:
                # Text-only response
                resp.message(response_dict['text'])
        
        else:
            resp.message("Please send a text message or image.")
        
        return str(resp)
    
    except Exception as e:
        print(f"Error processing webhook: {e}")
        import traceback
        traceback.print_exc()
        
        resp = MessagingResponse()
        resp.message("Sorry, I encountered an error. Please try again.")
        return str(resp)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "whatsapp-tuition-bot"}

@app.route("/", methods=["GET"])
def home():
    """Home page"""
    return """
    <h1>🎓 WhatsApp Tuition Bot</h1>
    <p>Bot is running!</p>
    <ul>
        <li>Webhook endpoint: /webhook</li>
        <li>Health check: /health</li>
    </ul>
    """

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 WhatsApp Tuition Bot Server")
    print("="*60)
    print(f"\n✅ Server starting...")
    print(f"📱 Twilio WhatsApp: {TWILIO_WHATSAPP_NUMBER}")
    print(f"\n🔗 Webhook URL (for Twilio): http://your-server.com/webhook")
    print(f"💚 Health check: http://your-server.com/health")
    print(f"\n{'='*60}\n")
    
    # Run Flask app
    app.run(
        host="0.0.0.0",  # Listen on all interfaces
        port=5000,
        debug=True
    )
