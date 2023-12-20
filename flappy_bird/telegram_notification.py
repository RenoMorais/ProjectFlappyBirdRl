from datetime import datetime

import requests

TOKEN = "<TELEGRAM_TOKEN_ID>"
CHAT_ID = "<TELEGRAM_CLIENT_ID>"

DATETIME_FORMAT = "%d/%m/%Y %H:%M:%S"


def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds

    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return hours, minutes, seconds


def begin_notification(start_date):
    send_notification(message=("Your training has started\n" f"Starting date: {start_date.strftime(DATETIME_FORMAT)}"))


def end_notification(start_date, failed):
    end_date = datetime.now()
    hours, minutes, seconds = convert_timedelta(end_date - start_date)

    duration_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"
    status_msg = "Failed!" if failed else "Success!"

    send_notification(
        message=(
            f"Training status: {status_msg}"
            f"\nEnd date: {end_date.strftime(DATETIME_FORMAT)}"
            f"\nTraining duration: {duration_str}"
        )
    )


def send_notification(message="Hello"):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
    r = requests.get(url)  # this sends the messageprint(requests.get(url).json()) # this sends the message
