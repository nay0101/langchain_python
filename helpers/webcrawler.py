import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Set, List, Optional
from .custom_types import _CRAWLING_TYPES
from langchain_community.document_loaders import WebBaseLoader


def get_page_content(url: str) -> Optional[str]:
    """
    Fetch the content of a given URL.

    Args:
    url (str): The URL to fetch.

    Returns:
    str: The HTML content of the page.

    Raises:
    requests.RequestException: If there's an error fetching the page.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_links(html_content: str, base_url: str) -> Set[str]:
    """
    Extract all links from the HTML content.

    Args:
    html_content (str): The HTML content to parse.
    base_url (str): The base URL to resolve relative links.

    Returns:
    set: A set of absolute URLs found in the HTML content.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    links = set()
    for a_tag in soup.find_all("a", href=True):
        link = urljoin(base_url, a_tag["href"])
        links.add(link)
    return links


def is_child_url(parent_url: str, child_url: str) -> bool:
    """
    Check if a URL is a child of the parent URL.

    Args:
    parent_url (str): The parent URL.
    child_url (str): The URL to check.

    Returns:
    bool: True if child_url is a child of parent_url, False otherwise.
    """
    parent_parsed = urlparse(parent_url)
    child_parsed = urlparse(child_url)

    # Check if domains match
    if parent_parsed.netloc != child_parsed.netloc:
        return False

    # Check if child path starts with parent path
    parent_path = parent_parsed.path.rstrip("/")
    child_path = child_parsed.path
    return child_path.startswith(parent_path + "/")


def crawl(
    start_url: str,
    crawling_method: _CRAWLING_TYPES = "crawl_child_urls",
    max_depth: int = 3,
    ignore_list: Optional[List[str]] = None,
) -> List[str]:
    """
    Crawl links within the same domain as the starting URL up to a maximum depth.

    Args:
    start_url (str): The starting URL for crawling.
    max_depth (int): The maximum depth to crawl (default is 3).

    Returns:
    set: A set of unique URLs crawled within the same domain.
    """
    visited = set()
    to_visit = [(start_url, 0)]
    start_domain = urlparse(start_url).netloc
    if ignore_list is None:
        ignore_list = []

    while to_visit:
        current_url, depth = to_visit.pop(0)
        if current_url in visited or current_url in ignore_list or depth > max_depth:
            continue

        if urlparse(current_url).netloc != start_domain:
            continue

        if (
            crawling_method == "crawl_child_urls"
            and depth > 0
            and not is_child_url(start_url, current_url)
        ):
            continue

        print(f"Crawling: {current_url}")
        visited.add(current_url)

        content = get_page_content(current_url)
        if content:
            links = extract_links(content, current_url)
            for link in links:
                if link not in visited and urlparse(link).netloc == start_domain:
                    to_visit.append((link, depth + 1))

    loader = WebBaseLoader(web_path=list(visited), requests_per_second=3)
    data = loader.load()
    return data
