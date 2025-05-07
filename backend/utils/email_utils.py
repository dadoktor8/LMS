import yagmail
import os
from dotenv import load_dotenv

load_dotenv()

def send_verification_email(to_email: str, link: str):
    user = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
   
    yag = yagmail.SMTP(user=user, password=password)
    subject = "Verify your email"
    content = f"Click the link to verify your email: {link}"
    yag.send(to=to_email, subject=subject, contents=content)
    print(f"✅ Email sent to {to_email}")
