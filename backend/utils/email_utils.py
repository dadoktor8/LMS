import yagmail
import os
import logging
import time
import random
import socket
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_verification_email(to_email: str, verification_link: str) -> bool:
    """
    Sends a verification email using Gmail SMTP + App Password.
    The email is styled to appear professional and less spammy.
    """
    email_user = os.environ.get("EMAIL_USER", "intellaica@gmail.com")
    email_password = os.environ.get("EMAIL_PASSWORD", "")
    sender_name = os.environ.get("SENDER_NAME", "Intellaica Team")
    company_name = os.environ.get("COMPANY_NAME", "Intellaica")

    if not email_user or not email_password:
        logger.error("Missing EMAIL_USER or EMAIL_PASSWORD in environment.")
        return False

    subject = f"Verify Your {company_name} Account"

    # A well-structured, mobile-friendly HTML with minimal spam triggers:
    html_content = f"""\

    <!DOCTYPE html> <html lang="en"> <head> <meta charset="utf-8"/> <meta name="viewport" content="width=device-width,initial-scale=1.0"/> <title>Verify Your {company_name} Account</title> <style> body {{ font-family: Tahoma, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; color: #333; line-height: 1.5; }} .container {{ max-width: 600px; background-color: #ffffff; margin: 40px auto; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }} .header {{ background-color: #0069d9; color: #fff; padding: 20px 30px; text-align: center; }} .header h1 {{ margin: 0; font-size: 24px; }} .content {{ padding: 30px; }} .cta-button {{ display: inline-block; background-color: #0069d9; color: #fff; text-decoration: none; padding: 12px 24px; border-radius: 4px; font-weight: bold; margin: 20px 0 30px 0; }} .footer {{ background-color: #f8f9fa; padding: 15px 30px; text-align: center; font-size: 12px; color: #6c757d; }} a.unsubscribe {{ color: #6c757d; text-decoration: underline; }} </style> </head> <body> <div class="container"> <div class="header"> <h1>{company_name}</h1> </div> <div class="content"> <h2>Verify Your Email Address</h2> <p>Hello,</p> <p> Thank you for creating a {company_name} account! Please click the button below to verify your email address and complete your registration. </p> <p style="text-align:center;"> <a href="{verification_link}" class="cta-button">Verify Email</a> </p> <p> If the button above doesn’t work, you can also verify by copying and pasting this link into your browser: </p> <p style="word-wrap:break-word; color:#0069d9;"> {verification_link} </p> <p><strong>This link expires in 1 hour.</strong></p> <p> If you did not request this, please ignore this email or contact our support team. </p> </div> <div class="footer"> <p> © 2024 {company_name}. All rights reserved.<br/> <a class="unsubscribe" href="mailto:{email_user}?subject=Unsubscribe">Unsubscribe</a> </p> </div> </div> </body> </html> """

    # Plain text fallback for better deliverability:
    text_content = f"""\

    {company_name} - Verify Your Account

    Hello,

    Please verify your email for {company_name} by clicking here:
    {verification_link}

    This link expires in 1 hour.

    If you did not request this, you can ignore this email.

    Best regards,
    The {company_name} Team
    © 2024 {company_name} - All rights reserved.
    To unsubscribe, reply with "Unsubscribe".
    """

    # Configure yagmail (Gmail: port=587 with TLS)
    try:
        socket.setdefaulttimeout(30)  # 30-second overall timeout
        yag = yagmail.SMTP(
            user=email_user,
            password=email_password,
            host="smtp.gmail.com",
            port=587,
            smtp_starttls=True,
            smtp_ssl=False  # For Gmail on port 587, use TLS (STARTTLS), not SSL
        )
        logger.info(f"Sending from {sender_name} <{email_user}> to {to_email} ...")
        yag.send(
            to=to_email,
            subject=subject,
            contents=html_content,
            headers={
                "From": f"{sender_name} <{email_user}>",
                "Reply-To": email_user,
                "X-Mailer": f"{company_name} Verification Service",
                "List-Unsubscribe": f"<mailto:{email_user}?subject=Unsubscribe>",
                "Precedence": "bulk",
                "Auto-Submitted": "auto-generated"
            }
        )
        logger.info("Verification email sent successfully!")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

