"""
anki_card_generator.py
Generate Chinese-learning Anki cards with an LLM and save them to ./output/cards.csv
Words are read from ./input.txt, separated by spaces.

Requirements:
  pip install openai tenacity
Environment:
  export OPENAI_API_KEY="sk-..."
"""

import csv
import json
import sys
from pathlib import Path
from time import sleep
from typing import List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ---------- config -----------------------------------------------------------

FINE_TUNED_MODEL = "ft:gpt-4o-mini-2025-04-18:personal::AoR7lm8l"
MODEL = FINE_TUNED_MODEL or "gpt-4o-mini"
CSV_PATH = Path("output/cards.csv")
RAW_LOG_PATH = Path("output/raw_cards_log.txt")
INPUT_PATH = Path("input.txt")
TEMPERATURE = 0.7
DELAY_BETWEEN_CALLS = 1  # seconds
CSV_DELIMITER = ";"      # safer than comma because sentences contain commas

# -----------------------------------------------------------------------------


def read_words(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        sys.exit(f"input file not found: {path.absolute()}")
    return [w.strip() for w in text.split() if w.strip()]


def build_prompt(word: str) -> str:
    return (
        "You are an Anki card maker for Chinese. "
        "Given a word or expression, return ONLY a JSON object with the keys:\n"
        "simplified, traditional, pinyin, translation, "
        "main_sentence, main_sentence_pinyin, main_sentence_english, "
        "sentences_battery, tag.\n\n"
        "Rules:\n"
        "• main_sentence must show the word in its core meaning.\n"
        "• sentences_battery is a single string containing 3-5 example "
        "sentences; separate each with a newline. Each sentence includes "
        "hanzi, pinyin, and English, divided by \" | \".\n"
        "• tag is one broad category (food, technology, classifier, travel, business, clothes, etc.).\n\n"
        f"Word: {word}"
    )


client = OpenAI()  # reads OPENAI_API_KEY from env


@retry(wait=wait_random_exponential(multiplier=1, max=20), stop=stop_after_attempt(5))
def fetch_card_json(word: str) -> str | None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an Anki card generator. "
                "Return ONLY valid JSON with the exact keys requested."
            ),
        },
        {"role": "user", "content": build_prompt(word)},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def main() -> None:
    words = read_words(INPUT_PATH)
    if not words:
        sys.exit("No words found in input.txt")

    CSV_PATH.parent.mkdir(exist_ok=True)

    with CSV_PATH.open("a", newline="", encoding="utf-8") as csvfile, RAW_LOG_PATH.open(
        "a", encoding="utf-8"
    ) as rawlog:
        fieldnames = [
            "simplified",
            "traditional",
            "pinyin",
            "translation",
            "main_sentence",
            "main_sentence_pinyin",
            "main_sentence_english",
            "sentences_battery",
            "tag",
        ]
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            delimiter=CSV_DELIMITER,
            quoting=csv.QUOTE_ALL,
        )

        for word in words:
            print(f"Processing '{word}' …")
            try:
                raw_json = fetch_card_json(word)
                rawlog.write(f"--- {word} ---\n{raw_json}\n\n")
                if not raw_json:
                    continue
                card = json.loads(raw_json)
                writer.writerow(card)
                sleep(DELAY_BETWEEN_CALLS)
            except Exception as exc:
                print(f"⚠️  skipped '{word}': {exc}")

    print("Done. Cards saved to", CSV_PATH.absolute())


if __name__ == "__main__":
    main()
