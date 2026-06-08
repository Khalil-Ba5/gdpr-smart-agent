import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Map article number ranges to GDPR chapter titles.
CHAPTERS = {
    range(1, 5): "Chapter I – General Provisions",
    range(5, 12): "Chapter II – Principles",
    range(12, 24): "Chapter III – Rights of the Data Subject",
    range(24, 44): "Chapter IV – Controller and Processor",
    range(44, 50): "Chapter V – Transfers to Third Countries",
    range(50, 60): "Chapter VI – Independent Supervisory Authorities",
    range(60, 77): "Chapter VII – Cooperation and Consistency",
    range(77, 85): "Chapter VIII – Remedies, Liability and Penalties",
    range(85, 92): "Chapter IX – Specific Processing Situations",
    range(92, 94): "Chapter X – Delegated Acts and Implementing Acts",
    range(94, 100): "Chapter XI – Final Provisions",
}

# Basic browser-like header to reduce the chance of being blocked.
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def get_chapter(article_number):
    """
    Return the GDPR chapter name for a given article number.
    """
    for article_range, chapter in CHAPTERS.items():
        if article_number in article_range:
            return chapter
    return "Unknown"


def get_full_text(paragraphs):
    """
    Combine all paragraph texts into one full article text string.

    Each paragraph is prefixed with its paragraph number so the
    original structure is still visible in the combined text.
    """
    return " ".join(f'{paragraph["number"]}. {paragraph["text"]}' for paragraph in paragraphs)


def scrape_article(article_number, session):
    """
    Scrape one GDPR article page and return a structured dictionary.

    Args:
        article_number (int): GDPR article number
        session (requests.Session): shared session for efficient requests

    Returns:
        dict: structured article data
    """
    url = f"https://gdpr-info.eu/art-{article_number}-gdpr/"

    # Fetch the page with a timeout so the script does not hang forever.
    response = session.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    # Parse the HTML into a searchable BeautifulSoup object.
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the page title safely in case the structure changes.
    title_span = soup.select_one("h1 .dsgvo-title")
    title = title_span.get_text(strip=True) if title_span else ""

    # Extract article paragraphs from the ordered list inside the main content.
    # stripped_strings is used instead of get_text(strip=True) to preserve spacing better.
    paragraphs = []

    # Select only the top-level <ol> inside entry-content
    top_ol = soup.select_one("div.entry-content > ol")
    if top_ol:
        # Get only direct <li> children, not nested ones
        top_level_items = top_ol.find_all("li", recursive=False)
        
        for i, item in enumerate(top_level_items, start=1):
            text = " ".join(item.stripped_strings)
            paragraphs.append({"number": i, "text": text})

    # Extract linked recital numbers shown near the article.
    recital_elements = soup.select("div.empfehlung-erwaegungsgruende span.bold-number")
    recitals = []

    for element in recital_elements:
        text = element.get_text(strip=True)

        # Some recital numbers may appear like "(12)", so remove parentheses first.
        text = text.replace("(", "").replace(")", "")

        if text.isdigit():
            recitals.append(int(text))

    # Remove duplicates while preserving original order.
    recitals = list(dict.fromkeys(recitals))

    # Build the final structured article object.
    article = {
        "article_number": article_number,
        "title": title,
        "chapter": get_chapter(article_number),
        "paragraphs": paragraphs,
        "full_text": get_full_text(paragraphs),
        "recitals": recitals,
        "url": url
    }

    return article


def scrape_articles(n):
    """
    Scrape GDPR articles from 1 to n and save them into a JSON file.
    """
    articles = []

    # Reuse one session across all requests for efficiency.
    with requests.Session() as session:
        for i in range(1, n + 1):
            try:
                article = scrape_article(i, session)
                articles.append(article)
                print(f"Scraped article {i}")
            except requests.RequestException as e:
                # Request-related failures are handled so one bad page
                # does not stop the whole scraping job.
                print(f"Failed to scrape article {i}: {e}")
            except Exception as e:
                # Catch unexpected parsing or logic errors as well.
                print(f"Unexpected error on article {i}: {e}")

    # Ensure the output directory exists before writing the file.
    output_dir = Path("data/docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "gdpr_articles.json"

    # Save the final list of articles as readable UTF-8 JSON.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(articles)} articles to {output_path}")


def main(n):
    """
    Entry point for the scraper.
    """
    scrape_articles(n)


if __name__ == "__main__":
    n = 99
    main(n)
