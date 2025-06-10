import yagmail
import boto3
import os
import logging
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        # SES Configuration
        self.ses_client = None
        self.ses_region = os.getenv("AWS_REGION", "us-east-2")
        self.sender_email = os.getenv("SENDER_EMAIL")
        
        # Yagmail Configuration (fallback)
        self.yagmail_user = os.getenv("EMAIL_USER")
        self.yagmail_password = os.getenv("EMAIL_PASSWORD")
        
        # Initialize SES client
        self._initialize_ses()
    
    def _initialize_ses(self):
        """Initialize Amazon SES client"""
        try:
            self.ses_client = boto3.client(
                'ses',
                region_name=self.ses_region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            logger.info("✅ Amazon SES client initialized successfully")
            print("✅ Amazon SES client initialized successfully")
        except NoCredentialsError:
            logger.warning("⚠️ AWS credentials not found. Will use yagmail as fallback.")
            print("⚠️ AWS credentials not found. Will use yagmail as fallback.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SES client: {str(e)}")
            print(f"❌ Failed to initialize SES client: {str(e)}")
    
    def _send_via_ses(self, to_email: str, subject: str, html_content: str, text_content: str) -> bool:
        """Send email via Amazon SES"""
        if not self.ses_client or not self.sender_email:
            print(f"❌ SES not available: client={bool(self.ses_client)}, sender={bool(self.sender_email)}")
            return False
        
        try:
            response = self.ses_client.send_email(
                Destination={'ToAddresses': [to_email]},
                Message={
                    'Body': {
                        'Html': {'Charset': 'UTF-8', 'Data': html_content},
                        'Text': {'Charset': 'UTF-8', 'Data': text_content}
                    },
                    'Subject': {'Charset': 'UTF-8', 'Data': subject}
                },
                Source=self.sender_email
            )
            logger.info(f"✅ Email sent via SES to {to_email}. Message ID: {response['MessageId']}")
            print(f"✅ Email sent via SES to {to_email}. Message ID: {response['MessageId']}")
            return True
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"❌ SES error sending to {to_email}: {error_code} - {error_message}")
            print(f"❌ SES error sending to {to_email}: {error_code} - {error_message}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error with SES: {str(e)}")
            print(f"❌ Unexpected error with SES: {str(e)}")
            return False
    
    def _send_via_yagmail(self, to_email: str, subject: str, html_content: str) -> bool:
        """Send email via yagmail (fallback) - simplified version"""
        if not self.yagmail_user or not self.yagmail_password:
            logger.error("❌ Yagmail credentials not configured")
            print("❌ Yagmail credentials not configured")
            return False
        
        try:
            print(f"🔄 Connecting to yagmail with user: {self.yagmail_user}")
            
            # Use your exact working code structure
            yag = yagmail.SMTP(user=self.yagmail_user, password=self.yagmail_password)
            
            # Extract verification link from HTML (same as your working version)
            verification_link = html_content.split('href="')[1].split('"')[0] if 'href="' in html_content else "Link not found"
            content = f"Click the link to verify your email: {verification_link}"
            
            print(f"📧 Sending email...")
            yag.send(to=to_email, subject=subject, contents=content)
            
            print(f"✅ Email sent to {to_email}")
            logger.info(f"✅ Email sent via yagmail to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email via yagmail: {str(e)}")
            print(f"❌ Failed to send email via yagmail: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            return False
    
    def send_verification_email(self, to_email: str, verification_link: str) -> bool:
        """Send professional verification email with SES/yagmail fallback"""
        subject = "Verify Your Email Address"
        
        # Simple HTML template with inline styles only
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f4;">
            <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px;">
                <h2 style="color: #333; text-align: center;">Verify Your Email Address</h2>
                <p style="color: #555; line-height: 1.6;">Thank you for signing up! Please click the button below to verify your email address and activate your account.</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{verification_link}" style="display: inline-block; background-color: #007bff; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">Verify Email Address</a>
                </div>
                <p style="color: #666; font-size: 14px;">If the button doesn't work, copy and paste this link into your browser:</p>
                <p style="color: #007bff; word-break: break-all; font-size: 14px;">{verification_link}</p>
                <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="color: #999; font-size: 12px; text-align: center;">This verification link will expire in 1 hour for security purposes.</p>
            </div>
        </body>
        </html>
        """
        
        # Plain text version for better deliverability
        text_content = f"""
        Welcome! Please verify your email address
        
        Thank you for creating an account. To complete your registration, please verify your email address by clicking the link below:
        
        {verification_link}
        
        This verification link will expire in 1 hour for your security.
        
        If you didn't create this account, please ignore this email.
        
        ---
        This is an automated message, please do not reply to this email.
        """
        
        # Try SES first, then fallback to yagmail
        print(f"🔍 Attempting to send email to {to_email}")
        print(f"🔍 SES Client available: {bool(self.ses_client)}")
        print(f"🔍 Sender email configured: {bool(self.sender_email)}")
        
        if self._send_via_ses(to_email, subject, html_content, text_content):
            return True
        
        logger.info("🔄 Falling back to yagmail...")
        print("🔄 Falling back to yagmail...")
        return self._send_via_yagmail(to_email, subject, html_content)

# Initialize global email service
email_service = EmailService()

def send_verification_email(to_email: str, link: str) -> bool:
    """Public function to send verification email"""
    return email_service.send_verification_email(to_email, link)