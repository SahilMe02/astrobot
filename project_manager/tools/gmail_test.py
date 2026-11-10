from __future__ import annotations

import os
from email.utils import parsedate_to_datetime

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TOKEN_FILE = os.path.join(BASE_DIR, "token.json")

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]


def get_header(headers: list[dict], name: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def main() -> None:
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(
            f"token.json not found at: {TOKEN_FILE}"
        )

    credentials = Credentials.from_authorized_user_file(
        TOKEN_FILE,
        SCOPES,
    )

    if not credentials.valid:
        raise RuntimeError(
            "Gmail credentials are not valid. "
            "Run gmail_auth.py again."
        )

    service = build(
        "gmail",
        "v1",
        credentials=credentials,
    )

    profile = (
        service.users()
        .getProfile(userId="me")
        .execute()
    )

    email_address = profile.get("emailAddress", "unknown")

    print(f"Gmail connection successful.")
    print(f"Authenticated account: {email_address}")
    print()

    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            maxResults=10,
        )
        .execute()
    )

    messages = response.get("messages", [])

    print(f"Messages found: {len(messages)}")
    print()

    if not messages:
        print("Mailbox is currently empty.")
        return

    for index, message in enumerate(messages, start=1):
        message_data = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message["id"],
                format="metadata",
                metadataHeaders=[
                    "Subject",
                    "From",
                    "Date",
                ],
            )
            .execute()
        )

        headers = message_data.get("payload", {}).get(
            "headers",
            [],
        )

        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        date_value = get_header(headers, "Date")

        formatted_date = date_value

        if date_value:
            try:
                formatted_date = parsedate_to_datetime(
                    date_value
                ).isoformat()
            except (TypeError, ValueError):
                pass

        print(f"--- Message {index} ---")
        print(f"Subject: {subject or '(no subject)'}")
        print(f"From: {sender or '(unknown sender)'}")
        print(f"Date: {formatted_date or '(unknown date)'}")
        print(f"Message ID: {message['id']}")
        print()


if __name__ == "__main__":
    main()