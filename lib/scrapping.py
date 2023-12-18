from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

headers = {
    'Accept': "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    'Accept-Encoding': "gzip, deflate",
    'Accept-Language': "en-US,en;q=0.5",
    # 'Alt-Used' and 'Host' will be set dynamically
    'Connection': "keep-alive",
    'Referer': "https://www.google.com/",
    'Sec-Fetch-Dest': "document",
    'Sec-Fetch-Mode': "navigate",
    'Sec-Fetch-Site': "cross-site",
    'Upgrade-Insecure-Requests': "1",
    'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0",
}


def get_html(base_url):
    domain = urlparse(base_url).hostname
    headers['Host'] = domain
    headers['Alt-Used'] = domain

    print(f"Query: {base_url}")
    try:
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
    except:
        # TODO: Enhance scrapper for javascript heavy pages
        print("Failed to fetch. Omitting this page")
        return ""

    content_type = response.headers['content-type'].split(';')[0]
    allowed_content_types = ["text/html", "application/json", "application/xml", "application/javascript", "text/plain"]

    if content_type not in allowed_content_types:
        # raise ValueError("Returned page was not utf8")
        print("Failed to fetch. Omitting this page")
        return ""

    return response.text


def get_text(html, base_url, summary):
    soup = BeautifulSoup(html, 'html.parser')
    root_element = 'body' if summary else ''
    text = []

    for elem in soup.select(f'{root_element}:not(style):not(script):not(svg)'):
        content = ''.join(elem.stripped_strings)
        if elem.name == 'a':
            href = elem.get('href')
            if href and not href.startswith('http'):
                href = urljoin(base_url, href)
            img_alt = elem.find('img', alt=True)
            if img_alt:
                content += f" {img_alt['alt']}"
            text.append(f' [{content}]({href})')
        elif content:
            text.append(content)

    return ' '.join(text).replace('\n', ' ')
