"""
WhatsApp Webhook with Cloudinary Image Hosting
==============================================

Handles incoming WhatsApp messages via Twilio and sends responses.
Images are uploaded to Cloudinary for delivery.

Author: Gotham AI
Version: 2.0 - Production with Cloudinary
"""

import os
import io
import hashlib
from datetime import datetime
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import bot functions
from tuition_bot import process_message, process_image

# ============================================
# CONFIGURATION
# ============================================
app = Flask(__name__)

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

# Cloudinary credentials
CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True
)

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# ============================================
# CLOUDINARY IMAGE UPLOAD
# ============================================
def upload_image_to_cloudinary(image_bytes: bytes, user_phone: str) -> str:
    """
    Upload image to Cloudinary and return public URL.
    
    Args:
        image_bytes: PNG image data
        user_phone: User's phone number (for organizing uploads)
        
    Returns:
        str: Public URL of uploaded image
    """
    try:
        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        phone_hash = hashlib.md5(user_phone.encode()).hexdigest()[:8]
        filename = f"math_solution_{phone_hash}_{timestamp}"
        
        # Upload to Cloudinary
        # Using 'tuition_bot' folder to organize images
        upload_result = cloudinary.uploader.upload(
            image_bytes,
            folder="tuition_bot",
            public_id=filename,
            resource_type="image",
            format="png",
            overwrite=True,
            # Auto-delete after 7 days to save space (optional)
            # Remove this line if you want to keep images forever
            tags=["math", "tuition", "temp"]
        )
        
        # Return secure URL
        return upload_result['secure_url']
        
    except Exception as e:
        print(f"❌ Cloudinary upload error: {e}")
        raise

# ============================================
# TWILIO MESSAGE SENDING
# ============================================
def send_whatsapp_message(to_phone: str, message: str, media_url: str = None):
    """
    Send WhatsApp message via Twilio.
    
    Args:
        to_phone: Recipient phone number (format: whatsapp:+1234567890)
        message: Text message to send
        media_url: Optional image URL to attach
    """
    try:
        # Format phone number
        if not to_phone.startswith('whatsapp:'):
            to_phone = f"whatsapp:{to_phone}"
        
        from_phone = f"whatsapp:{TWILIO_PHONE_NUMBER}"
        
        # Send message with optional media
        message_params = {
            'from_': from_phone,
            'to': to_phone,
            'body': message
        }
        
        if media_url:
            message_params['media_url'] = [media_url]
        
        msg = twilio_client.messages.create(**message_params)
        
        if DEBUG_MODE:
            print(f"✅ Message sent: {msg.sid}")
            if media_url:
                print(f"📸 Media attached: {media_url}")
        
        return msg.sid
        
    except Exception as e:
        print(f"❌ Twilio send error: {e}")
        raise

# ============================================
# WEBHOOK ROUTES
# ============================================
@app.route('/')
def home():
    """Health check endpoint"""
    return '''
    <html>
        <body style="font-family: Arial; padding: 40px; text-align: center;">
            <h1>🎓 WhatsApp Tuition Bot</h1>
            <p style="color: green; font-size: 18px;">✅ Server is running!</p>
            <p>Webhook endpoint: <code>/webhook</code></p>
            <hr style="margin: 40px 0;">
            <p style="color: gray; font-size: 12px;">
                Powered by OpenAI GPT-4 | Images via Cloudinary | Messages via Twilio
            </p>
        </body>
    </html>
    '''

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Main webhook endpoint for incoming WhatsApp messages.
    Handles both text and image messages, uploads math images to Cloudinary.
    """
    try:
        # Get incoming message data
        from_phone = request.values.get('From', '')
        message_body = request.values.get('Body', '').strip()
        num_media = int(request.values.get('NumMedia', 0))
        
        # Extract phone number (remove 'whatsapp:' prefix)
        phone = from_phone.replace('whatsapp:', '')
        
        if DEBUG_MODE:
            print("=" * 60)
            print(f"📱 Incoming from: {phone}")
            print(f"📝 Message: {message_body}")
            print(f"🖼️  Media: {num_media}")
            print("=" * 60)
        
        # Process message
        if num_media > 0:
            # Handle image message
            media_url = request.values.get('MediaUrl0')
            media_content_type = request.values.get('MediaContentType0', '')
            
            if DEBUG_MODE:
                print(f"Processing image: {media_url}")
                print(f"Image format: {media_content_type}")
            
            # Download image with Twilio authentication
            import requests
            from requests.auth import HTTPBasicAuth
            
            img_response = requests.get(
                media_url,
                auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            )
            image_bytes = img_response.content
            
            if DEBUG_MODE:
                print(f"Downloaded {len(image_bytes)} bytes")
                print(f"Download status: {img_response.status_code}")
            
            # Process with bot
            response = process_image(phone, image_bytes, message_body)
        else:
            # Handle text message
            if DEBUG_MODE:
                print("Processing text message...")
            
            response = process_message(phone, message_body)
        
        # Send response
        if 'image_bytes' in response:
            # Bot generated a math image - upload to Cloudinary
            if DEBUG_MODE:
                print("📐 Uploading math solution image to Cloudinary...")
            
            try:
                # Upload to Cloudinary
                image_url = upload_image_to_cloudinary(
                    response['image_bytes'], 
                    phone
                )
                
                if DEBUG_MODE:
                    print(f"✅ Image uploaded: {image_url}")
                
                # Send message with image
                send_whatsapp_message(
                    phone,
                    response['text'],
                    media_url=image_url
                )
                
            except Exception as e:
                # If image upload fails, send text-only response
                print(f"⚠️  Image upload failed, sending text only: {e}")
                send_whatsapp_message(phone, response['text'])
        else:
            # Text-only response
            send_whatsapp_message(phone, response['text'])
        
        # Return empty TwiML response (we're sending via API)
        return '', 200
        
    except Exception as e:
        print(f"❌ Webhook error: {e}")
        import traceback
        traceback.print_exc()
        
        # Send error message to user
        try:
            send_whatsapp_message(
                phone,
                "Sorry, I encountered an error. Please try again in a moment."
            )
        except:
            pass
        
        return '', 500

@app.route('/health', methods=['GET'])
def health():
    """Health check for monitoring"""
    return {
        'status': 'healthy',
        'service': 'whatsapp-tuition-bot',
        'cloudinary': 'configured' if CLOUDINARY_CLOUD_NAME else 'not configured',
        'twilio': 'configured' if TWILIO_ACCOUNT_SID else 'not configured'
    }, 200

# ============================================
# STARTUP
# ============================================
if __name__ == '__main__':
    # Check required environment variables
    required_vars = [
        'TWILIO_ACCOUNT_SID',
        'TWILIO_AUTH_TOKEN', 
        'TWILIO_PHONE_NUMBER',
        'CLOUDINARY_CLOUD_NAME',
        'CLOUDINARY_API_KEY',
        'CLOUDINARY_API_SECRET',
        'OPENAI_API_KEY'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("⚠️  WARNING: Missing environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nSet these in your Render dashboard or .env file")
    else:
        print("✅ All environment variables configured")
    
    print("\n" + "="*60)
    print("🎓 WhatsApp Tuition Bot - Starting Server")
    print("="*60)
    print(f"📡 Cloudinary: {CLOUDINARY_CLOUD_NAME or 'NOT CONFIGURED'}")
    print(f"📱 Twilio: {TWILIO_PHONE_NUMBER or 'NOT CONFIGURED'}")
    print(f"🐛 Debug mode: {DEBUG_MODE}")
    print("="*60 + "\n")
    
    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG_MODE)
