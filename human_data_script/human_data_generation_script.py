import requests
import time
import json
import re
from typing import List, Dict, Optional, Set
from datetime import datetime

# 1. WIKIPEDIA API COLLECTOR

class WikipediaCollector:

    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-Detection-Research/1.0 (Educational Purpose)'
        })

    def get_articles_by_category(self, category: str,
                                 num_articles: int = 50,
                                 continue_token: Optional[str] = None) -> tuple[List[str], Optional[str]]:
        """
        Get article titles from a specific Wikipedia category
        Returns: (titles, next_continue_token)
        """
        titles = []

        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category}',
            'cmlimit': min(num_articles, 50),  # API max is 50
            'cmtype': 'page'
        }

        if continue_token:
            params['cmcontinue'] = continue_token

        try:
            response = self.session.get(self.base_url, params=params)
            data = response.json()

            for page in data['query']['categorymembers']:
                titles.append(page['title'])

            # Get continuation token if available
            next_token = None
            if 'continue' in data:
                next_token = data['continue']['cmcontinue']

            print(f"Fetched {len(titles)} articles from '{category}'...")
            time.sleep(0.5)

            return titles, next_token

        except Exception as e:
            print(f"Error getting category articles: {e}")
            return [], None

    def get_article_content(self, title: str,
                           extract_sections: bool = True) -> Optional[Dict]:
        """
        Get the full content of a Wikipedia article
        """
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|info',
            'explaintext': True,
            'exsectionformat': 'plain' if extract_sections else 'raw',
            'inprop': 'url'
        }

        try:
            response = self.session.get(self.base_url, params=params)
            data = response.json()

            pages = data['query']['pages']
            page_id = list(pages.keys())[0]

            if page_id == '-1':
                print(f"Article '{title}' not found")
                return None

            page = pages[page_id]

            return {
                'title': page.get('title'),
                'content': page.get('extract', ''),
                'url': page.get('fullurl', ''),
                'page_id': page_id
            }

        except Exception as e:
            print(f"Error getting article '{title}': {e}")
            return None


# 2. TEXT PROCESSING AND CHUNKING


def clean_wikipedia_text(text: str) -> str:
    """
    Clean Wikipedia text to remove metadata and formatting
    """
    if not text:
        return ""

    sections_to_remove = [
        r'\n==\s*See also\s*==.*',
        r'\n==\s*References\s*==.*',
        r'\n==\s*External links\s*==.*',
        r'\n==\s*Notes\s*==.*',
        r'\n==\s*Further reading\s*==.*'
    ]

    for pattern in sections_to_remove:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(r'\n=+\s*.*?\s*=+\n', '\n', text)
    text = re.sub(r'\[\d+\]|\[citation needed\]|\[clarification needed\]', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


def chunk_text(text: str,
               chunk_size: int = 200,
               min_words: int = 80,
               overlap: int = 40) -> List[str]:
    words = text.split()
    chunks = []

    for start in range(0, len(words), chunk_size - overlap):
        chunk = words[start:start + chunk_size]
        if len(chunk) >= min_words:
            chunks.append(" ".join(chunk))

    return chunks


# 3. DATASET CREATION WITH DEDUPLICATION


def collect_wikipedia_dataset(num_samples: int = 2000,
                              domains: Optional[List[str]] = None,
                              target_words: int = 300,
                              overlap: int = 40) -> List[Dict]:
    """
    Collect Wikipedia dataset from specific categories with deduplication

    Args:
        num_samples: Number of text samples to collect
        domains: List of Wikipedia categories to collect from
        target_words: Target word count per sample
        overlap: Word overlap between chunks (set to 0 to avoid chunk overlap)
    """

    collector = WikipediaCollector()
    samples = []

    # Track processed articles to avoid duplication
    processed_articles: Set[str] = set()

    # Track continuation tokens for each domain
    domain_tokens: Dict[str, Optional[str]] = {}

    if domains is None:
        domains = [
            "Science", "History", "Technology", "Biology", "Physics",
            'Geography', 'Literature', 'Philosophy', "Mathematics",
            "Computer Science", "Medicine", "Economics", "Psychology",
            'climate change', 'world war', 'constitution',
            'quantum mechanics', 'human evolution', 'renewable energy',
            'ancient civilization', 'modern art', 'space exploration',
            "Artificial Intelligence", "The Industrial Revolution", "Quantum Mechanics",
            "AStronomy", "Environmental Science", "Machine Learning",
            "Renewable Energy", "Genetics", "Space Exploration", "Blockchain","Animals and Zoology",
            "Engineering","Education"
        ]

    print(f"Starting Wikipedia collection: {num_samples} samples")
    print(f"Method: Category-based with deduplication")
    print(f"Domains: {len(domains)} categories")
    print(f"Overlap: {overlap} words")
    print("=" * 60)

    domain_index = 0
    max_attempts = num_samples * 3  # Safety limit
    attempts = 0

    while len(samples) < num_samples and attempts < max_attempts:
        domain = domains[domain_index % len(domains)]

        # Get continuation token for this domain
        continue_token = domain_tokens.get(domain)

        print(f"\nCollecting from category: {domain} (attempt {attempts + 1})")

        titles, next_token = collector.get_articles_by_category(
            domain,
            num_articles=50,
            continue_token=continue_token
        )

        # Update continuation token
        domain_tokens[domain] = next_token

        # If no new articles from this domain, move to next
        if not titles:
            print(f"No more articles in '{domain}', moving to next domain...")
            domain_index += 1
            continue

        new_articles_processed = 0

        for title in titles:
            if len(samples) >= num_samples:
                break

            # Skip if already processed
            if title in processed_articles:
                print(f"Skipping duplicate: {title}")
                continue

            processed_articles.add(title)

            article = collector.get_article_content(title)
            if not article or not article['content']:
                continue

            processed_samples = process_article(
                article,
                domain=domain.lower(),
                target_words=target_words,
                overlap=overlap
            )

            if not processed_samples:
                continue

            samples.extend(processed_samples)
            new_articles_processed += 1

            if len(samples) >= num_samples:
                break

            time.sleep(0.3)

        print(f"New articles processed: {new_articles_processed}")
        print(f"Total samples collected: {len(samples)}/{num_samples}")
        print(f"Unique articles processed: {len(processed_articles)}")

        attempts += 1
        domain_index += 1

    # Trim to exact number requested
    samples = samples[:num_samples]

    print(f"\n{'=' * 60}")
    print(f"✓ Collection complete!")
    print(f"Total samples collected: {len(samples)}")
    print(f"Unique articles used: {len(processed_articles)}")
    print(f"Average chunks per article: {len(samples) / len(processed_articles):.2f}")

    return samples


def process_article(article: Dict, domain: str,
                   target_words: int = 300,
                   overlap: int = 40) -> List[Dict]:
    """
    Process a single Wikipedia article into multiple samples
    """

    cleaned_text = clean_wikipedia_text(article['content'])
    word_count = len(cleaned_text.split())

    # Skip very short / stub articles
    if word_count < 800:
        return []

    # Chunk the full article
    chunks = chunk_text(
        cleaned_text,
        chunk_size=target_words,
        min_words=int(target_words * 0.6),
        overlap=overlap
    )

    samples = []
    for i, text in enumerate(chunks):
        samples.append({
            'id': f"wiki_{article['page_id']}_{i}",
            'text': text,
            'label': 0,
            'domain': domain,
            'source': 'wikipedia',
            'article_title': article['title'],
            'url': article['url'],
            'word_count': len(text.split()),
            'collection_date': datetime.now().isoformat(),
            'modifications': 'none',
            'chunk_index': i
        })

    return samples


# 4. SAVE AND EXPORT

def save_wikipedia_dataset(samples: List[Dict], output_dir: str = 'wikipedia_data'):
    """
    Save collected Wikipedia dataset
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON
    with open(f'{output_dir}/human_data.json', 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    # Generate detailed statistics
    stats = {
        'total_samples': len(samples),
        'unique_articles': len(set(s['article_title'] for s in samples)),
        'avg_word_count': sum(s['word_count'] for s in samples) / len(samples),
        'min_word_count': min(s['word_count'] for s in samples),
        'max_word_count': max(s['word_count'] for s in samples),
        'domains': {},
        'collection_date': datetime.now().isoformat()
    }

    # Count by domain
    for sample in samples:
        domain = sample.get('domain', 'unknown')
        stats['domains'][domain] = stats['domains'].get(domain, 0) + 1

    # Calculate chunks per article
    article_chunks = {}
    for sample in samples:
        article = sample['article_title']
        article_chunks[article] = article_chunks.get(article, 0) + 1

    stats['avg_chunks_per_article'] = sum(article_chunks.values()) / len(article_chunks)
    stats['max_chunks_from_single_article'] = max(article_chunks.values())

    with open(f'{output_dir}/human_data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n Dataset saved to {output_dir}/")
    print(f"  - human_data.json")
    print(f"  - human_data_stats.json")

    return df


# 5. EXAMPLE USAGE


if __name__ == "__main__":
    print("Wikipedia Human Text Collector - Category-Based with Deduplication")
    print("=" * 60)

    # Collect 2000 samples from specific categories
    samples = collect_wikipedia_dataset(
        num_samples=2000,
        domains=[
            "Science", "History", "Technology", "Biology", "Physics",
            'Geography', 'Literature', 'Philosophy', "Mathematics",
            "Computer Science", "Medicine", "Economics", "Psychology",
            'climate change', 'world war', 'constitution',
            'quantum mechanics', 'human evolution', 'renewable energy',
            'ancient civilization', 'modern art', 'space exploration',
            "Artificial Intelligence", "The Industrial Revolution", "Quantum Mechanics",
            "AStronomy", "Environmental Science", "Machine Learning",
            "Renewable Energy", "Genetics", "Space Exploration", "Blockchain","Animals and Zoology",
            "Engineering","Education",
        ],
        target_words=300,
        overlap=40  # Set to 0 for no overlap between chunks
    )

    # Save dataset
    df = save_wikipedia_dataset(samples, output_dir='dataset/human_data')

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"Total samples: {len(samples)}")
    print(f"Unique articles: {df['article_title'].nunique()}")
    print(f"Average word count: {df['word_count'].mean():.1f}")
    print("\nDomain distribution:")
    print(df['domain'].value_counts())
