import mailtrap as mt
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

mail = mt.Mail(
    sender=mt.Address(email="hello@demomailtrap.com", name="Mailtrap Test"),
    to=[mt.Address(email=os.getenv('EMAIL_TO'))],
    subject="You are awesome!",
    text="Congrats for sending test email with Mailtrap!",
    category="Integration Test",
)

client = mt.MailtrapClient(token=os.getenv('MAILTRAP_TOKEN'))
response = client.send(mail)

print(response)