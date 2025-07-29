"""
Gundam GCG Card Scraper
=======================

A comprehensive scraper for the official Gundam GCG (Gundam Card Game) website.
Downloads all card metadata, artwork, and set information from https://www.gundam-gcg.com.

Features:
- Scrapes all card sets (base, parallels, promos, future releases)
- Downloads high-resolution card artwork
- Extracts structured metadata (stats, text, keywords, etc.)
- Supports multiple languages (en, jp, zh-tw, etc.)
- Parallel processing for efficient downloads
- Resumable downloads (skips existing images)
- Outputs JSON and CSV formats

Usage:
    python scrape_gcg.py [--lang en] [--max-conn 16] [--skip-images]

Arguments:
    --lang         Language to scrape (default: en)
    --max-conn     Maximum parallel connections (default: 16)
    --skip-images  Skip artwork downloads, metadata only

Outputs:
    - gundam_cards.json   - Complete card data with Unicode preserved
    - gundam_cards.csv    - Excel-friendly flat file
    - card_images/        - High-res artwork (one JPG per card)

Requirements:
    Python >= 3.9
    aiohttp
    beautifulsoup4
    lxml
    tqdm

Author: Open Source Contributor
License: MIT
"""

import asyncio
import aiohttp
import csv
import json
import pathlib
import re
import argparse
import logging
from typing import Dict, Set, List, Optional, Tuple, Any
from dataclasses import dataclass

from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_DOMAIN = "https://www.gundam-gcg.com"
CARD_IMG_DIR = pathlib.Path("card_images")
DEFAULT_TIMEOUT = 25
DEFAULT_MAX_CONNECTIONS = 16

# Regex patterns for parsing
DETAIL_URL_PATTERN = re.compile(r"detail\.php\?detailSearch=([A-Z0-9\-_p]+)")
SET_CODE_PATTERN = re.compile(r"\[([A-Z0-9]+)]")
PILOT_PATTERN = re.compile(r'\[Pilot\]\[([^\]]+)\]')
ACTION_TYPE_PATTERN = re.compile(r'\[([^\]]+)\]')
KEYWORD_PATTERN = re.compile(r'<([^>]+)>')


@dataclass
class SetInfo:
    """Represents information about a card set."""
    name: str
    package_code: str
    url: str


@dataclass
class CardData:
    """Represents structured card data."""
    card_id: str
    card_number: str
    name: str
    level: str
    cost: str
    color: str
    card_type: str
    ap: str
    hp: str
    text: str
    zone: str
    trait: str
    link: str
    rarity: str
    source: str
    set_name: str
    where_to_get_it: str
    image_url: str
    pilot: str
    keywords: List[str]


class GCGScraper:
    """
    Main scraper class for Gundam GCG website.
    
    Handles all scraping operations including set discovery, card parsing,
    and image downloading with proper error handling and rate limiting.
    """
    
    def __init__(self, lang: str = "en", max_connections: int = DEFAULT_MAX_CONNECTIONS):
        """
        Initialize the scraper.
        
        Args:
            lang: Language code for the website (e.g., 'en', 'jp', 'zh-tw')
            max_connections: Maximum parallel HTTP connections
        """
        self.lang = lang
        self.max_connections = max_connections
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit_per_host=self.max_connections)
        headers = {
            "User-Agent": "Mozilla/5.0 (GCG-Scraper/1.0)",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        self.session = aiohttp.ClientSession(
            connector=connector, 
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def _build_root_url(self) -> str:
        """Build the URL for the main card list page."""
        return f"{BASE_DOMAIN}/{self.lang}/cards/index.php"

    def _build_detail_url(self, card_id: str) -> str:
        """Build the URL for a specific card's detail page."""
        return f"{BASE_DOMAIN}/{self.lang}/cards/detail.php?detailSearch={card_id}"

    async def _fetch(self, url: str, as_bytes: bool = False) -> str | bytes:
        """
        Fetch content from a URL with error handling.
        
        Args:
            url: URL to fetch
            as_bytes: Whether to return bytes instead of text
            
        Returns:
            Response content as string or bytes
            
        Raises:
            aiohttp.ClientError: If the request fails
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
            
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                if as_bytes:
                    return await response.read()
                else:
                    return await response.text()
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise

    async def discover_sets(self) -> Dict[str, SetInfo]:
        """
        Discover all available card sets from the main page.
        
        Returns:
            Dictionary mapping set codes to SetInfo objects
        """
        logger.info("Discovering card sets...")
        html = await self._fetch(self._build_root_url())
        soup = BeautifulSoup(html, "lxml")
        
        sets: Dict[str, SetInfo] = {}
        
        # Parse set links with the new structure
        for link in soup.select("a.js-selectBtn-package[data-val]"):
            value = link.get("data-val", "")
            if isinstance(value, list):
                value = value[0] if value else ""
            value = value.strip()
            label = link.get_text(strip=True)
            
            # Skip the "ALL" option and empty values
            if not value or value == "" or label == "ALL" or label == "Included In：ALL":
                continue
                
            # Extract set code from brackets
            set_match = SET_CODE_PATTERN.search(label)
            if set_match:
                # Has set code - use the code as the key
                set_code = set_match.group(1)
                set_name = label.split("[")[0].strip()
            else:
                # Handle promotional cards without set codes
                if any(promo_indicator in label.lower() 
                       for promo_indicator in ['challenge', 'promotion', 'promo', 'event', 'tournament']):
                    set_code = "PROMO"
                    set_name = "Promotion Card"
                else:
                    # Fallback for other non-bracketed sets
                    set_code = f"PKG{value}"
                    set_name = label.strip()
            
            sets[set_code] = SetInfo(
                name=set_name,
                package_code=value,
                url=f"{BASE_DOMAIN}/{self.lang}/cards/index.php?package={value}"
            )

        # Fallback: regex parsing if DOM structure changes
        if not sets:
            logger.warning("Using fallback regex parsing for sets")
            pattern = re.compile(r'data-val="(\d+)"[^>]*>([^<]+?)\s*\[([A-Z0-9]+)]', re.I)
            html_str = html if isinstance(html, str) else html.decode()
            for pkg, label, code in pattern.findall(html_str):
                sets[code] = SetInfo(
                    name=label.strip(),
                    package_code=pkg,
                    url=f"{BASE_DOMAIN}/{self.lang}/cards/index.php?package={pkg}"
                )
        
        logger.info(f"Discovered {len(sets)} sets")
        return sets

    async def gather_card_urls(self, set_info: SetInfo, set_code: str) -> Set[Tuple[str, str]]:
        """
        Gather all card detail URLs from a set page.
        
        Args:
            set_info: Information about the set
            set_code: The set code identifier
            
        Returns:
            Set of tuples containing (detail_url, set_code)
        """
        html = await self._fetch(set_info.url)
        # Use regex for faster parsing and resilience to markup changes
        if isinstance(html, str):
            card_ids = set(DETAIL_URL_PATTERN.findall(html))
        else:
            card_ids = set(DETAIL_URL_PATTERN.findall(html.decode()))
        return {(self._build_detail_url(card_id), set_code) for card_id in card_ids}

    def _extract_card_data(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract card data from the detail page HTML.
        
        Args:
            soup: BeautifulSoup object of the card detail page
            
        Returns:
            Dictionary of card attributes
        """
        data = {}
        
        # Parse the new dl/dt/dd structure
        for dl in soup.select("dl.dataBox"):
            dt = dl.select_one("dt.dataTit")
            dd = dl.select_one("dd.dataTxt")
            if dt and dd:
                key = dt.get_text(strip=True)
                value = dd.get_text(" ", strip=True)
                data[key] = self._clean_text(value)
        
        # Extract card text from overview section
        overview = soup.select_one(".cardDataRow.overview .dataTxt")
        if overview:
            text_content = overview.get_text("\n", strip=True)
            data["Text"] = self._clean_text(text_content)
        
        return data

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with normalized whitespace and brackets
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Normalize whitespace
        cleaned = " ".join(text.split())
        
        # Fix spacing around Japanese brackets
        cleaned = re.sub(r'\s+【', r'【', cleaned)
        cleaned = re.sub(r'\s+】', r'】', cleaned)
        cleaned = re.sub(r'\s+（', r'（', cleaned)
        cleaned = re.sub(r'\s+）', r'）', cleaned)
        cleaned = re.sub(r'【\s+', r'【', cleaned)
        cleaned = re.sub(r'（\s+', r'（', cleaned)
        
        # Convert Japanese brackets to English
        cleaned = cleaned.replace('【', '[').replace('】', ']')
        
        return cleaned

    def _extract_card_number(self, card_id: str) -> str:
        """
        Extract base card number by removing parallel indicators.
        
        Args:
            card_id: Full card ID (may include parallel indicators)
            
        Returns:
            Base card number without parallel indicators
        """
        # Remove parallel part (e.g., _p1, _p2, etc.)
        if '_p' in card_id:
            return card_id.split('_p')[0]
        return card_id

    def _extract_action_types(self, text: str) -> List[str]:
        """
        Extract action types from square brackets in card text.
        
        Args:
            text: Card text to parse
            
        Returns:
            List of action types found
        """
        action_types = set()
        
        # Temporarily replace pilot patterns to avoid capturing them
        temp_text = re.sub(r'\[Pilot\]\[([^\]]+)\]', r'[PILOT_PLACEHOLDER]', text)
        matches = ACTION_TYPE_PATTERN.findall(temp_text)
        
        for match in matches:
            action_type = match.strip()
            if action_type and action_type != "PILOT_PLACEHOLDER":
                action_types.add(action_type)
                
        return list(action_types)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from angle brackets in card text.
        
        Args:
            text: Card text to parse
            
        Returns:
            List of keywords found
        """
        keywords = set()
        matches = KEYWORD_PATTERN.findall(text)
        
        for match in matches:
            keyword = match.strip()
            if keyword:
                keywords.add(keyword)
                # Also add base keyword (first word) if different
                base_keyword = keyword.split()[0] if ' ' in keyword else None
                if base_keyword and base_keyword != keyword:
                    keywords.add(base_keyword)
                    
        return list(keywords)

    def _extract_pilot_name(self, text: str) -> str:
        """
        Extract pilot name from [Pilot][Name] pattern.
        
        Args:
            text: Card text to parse
            
        Returns:
            Pilot name if found, empty string otherwise
        """
        pilot_match = PILOT_PATTERN.search(text)
        if pilot_match:
            return pilot_match.group(1).strip()
        return ""

    async def _download_image(self, image_url: str, card_id: str) -> bool:
        """
        Download card image if it doesn't already exist.
        
        Args:
            image_url: URL of the image to download
            card_id: Card ID for filename
            
        Returns:
            True if download successful or already exists, False otherwise
        """
        img_path = CARD_IMG_DIR / f"{card_id}.webp"
        
        # Skip if image already exists
        if img_path.exists():
            return True
            
        try:
            # Ensure directory exists
            CARD_IMG_DIR.mkdir(parents=True, exist_ok=True)
            
            # Download image
            img_bytes = await self._fetch(image_url, as_bytes=True)
            if isinstance(img_bytes, str):
                img_bytes = img_bytes.encode()
                
            img_path.write_bytes(img_bytes)
            return True
            
        except Exception as e:
            logger.error(f"Failed to download image for {card_id}: {e}")
            return False

    async def parse_card_detail(self, url: str, set_code: str, download_images: bool = True) -> Optional[CardData]:
        """
        Parse a single card detail page and extract all metadata.
        
        Args:
            url: URL of the card detail page
            set_code: Set code the card belongs to
            download_images: Whether to download card artwork
            
        Returns:
            CardData object if successful, None if failed
        """
        try:
            card_id = url.split("detailSearch=")[1]
            html = await self._fetch(url)
            soup = BeautifulSoup(html, "lxml")

            # Extract image URL and download if requested
            image_url = ""
            card_img_tag = soup.select_one(".cardImage img")
            if card_img_tag and card_img_tag.has_attr('src'):
                img_src = card_img_tag['src']
                if isinstance(img_src, list):
                    img_src = img_src[0] if img_src else ""
                    
                if img_src.startswith('../'):
                    # Convert relative URL to absolute
                    image_url = f"{BASE_DOMAIN}/{self.lang}{img_src[2:]}"
                else:
                    image_url = img_src
                    
                if download_images and image_url:
                    await self._download_image(image_url, card_id)

            # Extract structured data
            card_data = self._extract_card_data(soup)
            
            # Extract name and rarity
            name_node = soup.select_one("h1.cardName")
            rarity_node = soup.select_one("div.rarity")
            
            name = self._clean_text(name_node.get_text(strip=True)) if name_node else ""
            rarity = self._clean_text(rarity_node.get_text(strip=True)) if rarity_node else ""
            
            # Extract and clean text
            text = self._clean_text(card_data.get("Text", ""))
            
            # Extract tags and keywords
            action_types = self._extract_action_types(text)
            keywords = self._extract_keywords(text)
            pilot_name = self._extract_pilot_name(text)
            all_keywords = list(set(action_types + keywords))
            
            return CardData(
                card_id=card_id,
                card_number=self._extract_card_number(card_id),
                name=name,
                level=self._clean_text(card_data.get("Lv.", "")),
                cost=self._clean_text(card_data.get("COST", "")),
                color=self._clean_text(card_data.get("COLOR", "")),
                card_type=self._clean_text(card_data.get("TYPE", "")),
                ap=self._clean_text(card_data.get("AP", "")),
                hp=self._clean_text(card_data.get("HP", "")),
                text=text,
                zone=self._clean_text(card_data.get("Zone", "")),
                trait=self._clean_text(card_data.get("Trait", "")),
                link=self._clean_text(card_data.get("Link", "")),
                rarity=rarity,
                source=self._clean_text(card_data.get("Source Title", "")),
                set_name="Promotion Card" if set_code == "PROMO" 
                         else self._clean_text(card_data.get("Where to get it", "")),
                where_to_get_it=self._clean_text(card_data.get("Where to get it", "")),
                image_url=image_url,
                pilot=pilot_name,
                keywords=all_keywords
            )
            
        except Exception as e:
            logger.error(f"Failed to parse card at {url}: {e}")
            return None

    def _card_data_to_dict(self, card: CardData) -> Dict[str, Any]:
        """
        Convert CardData object to dictionary for JSON/CSV output.
        
        Args:
            card: CardData object to convert
            
        Returns:
            Dictionary representation of the card
        """
        return {
            "card_id": card.card_id,
            "card_number": card.card_number,
            "name": card.name,
            "level": card.level,
            "cost": card.cost,
            "color": card.color,
            "type": card.card_type,
            "ap": card.ap,
            "hp": card.hp,
            "text": card.text,
            "zone": card.zone,
            "trait": card.trait,
            "link": card.link,
            "rarity": card.rarity,
            "source": card.source,
            "set": card.set_name,
            "where_to_get_it": card.where_to_get_it,
            "image": card.image_url,
            "pilot": card.pilot,
            "keywords": card.keywords,
        }

    async def scrape_all_cards(self, download_images: bool = True) -> List[Dict[str, Any]]:
        """
        Main scraping method that orchestrates the entire process.
        
        Args:
            download_images: Whether to download card artwork
            
        Returns:
            List of card dictionaries ready for output
        """
        # Step 1: Discover all sets
        sets = await self.discover_sets()
        logger.info(f"Found {len(sets)} sets: {', '.join(sorted(sets.keys()))}")

        # Step 2: Gather all card URLs
        all_card_urls: Set[Tuple[str, str]] = set()
        for set_code, set_info in sets.items():
            card_urls = await self.gather_card_urls(set_info, set_code)
            all_card_urls.update(card_urls)
            
        logger.info(f"Found {len(all_card_urls)} unique card detail pages")

        # Step 3: Parse all cards concurrently
        semaphore = asyncio.Semaphore(self.max_connections)

        async def parse_card_with_semaphore(url_and_set: Tuple[str, str]) -> Optional[CardData]:
            """Parse a single card with semaphore for rate limiting."""
            url, set_code = url_and_set
            async with semaphore:
                return await self.parse_card_detail(url, set_code, download_images)

        # Create tasks for all cards
        tasks = [parse_card_with_semaphore(url_and_set) for url_and_set in all_card_urls]
        
        # Execute with progress bar
        cards = []
        results = await tqdm_asyncio.gather(*tasks, ncols=88)
        for result in results:
            if result:
                cards.append(self._card_data_to_dict(result))

        logger.info(f"Successfully parsed {len(cards)} cards")
        return cards

    def save_outputs(self, cards: List[Dict[str, Any]]) -> None:
        """
        Save scraped data to JSON and CSV files.
        
        Args:
            cards: List of card dictionaries to save
        """
        # Save JSON with Unicode preservation
        with open("gundam_cards.json", "w", encoding="utf-8") as f:
            json.dump(cards, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(cards)} cards to gundam_cards.json")

        # Save CSV for Excel compatibility
        if cards:
            with open("gundam_cards.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cards[0].keys())
                writer.writeheader()
                writer.writerows(cards)
            logger.info(f"Saved {len(cards)} cards to gundam_cards.csv")


async def main():
    """
    Main entry point for the scraper.
    
    Handles command line arguments and orchestrates the scraping process.
    """
    parser = argparse.ArgumentParser(
        description="Download & dump all Gundam GCG cards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_gcg.py                    # Scrape English cards with images
  python scrape_gcg.py --lang jp         # Scrape Japanese cards
  python scrape_gcg.py --skip-images     # Metadata only, no images
  python scrape_gcg.py --max-conn 8      # Use fewer connections
        """
    )
    
    parser.add_argument(
        "--lang", 
        default="en", 
        help="Site language (default: en)"
    )
    parser.add_argument(
        "--max-conn", 
        type=int, 
        default=DEFAULT_MAX_CONNECTIONS,
        help=f"Parallel connections (default: {DEFAULT_MAX_CONNECTIONS})"
    )
    parser.add_argument(
        "--skip-images", 
        action="store_true", 
        help="Skip downloading artwork"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Gundam GCG scraper")
    logger.info(f"Language: {args.lang}")
    logger.info(f"Max connections: {args.max_conn}")
    logger.info(f"Download images: {not args.skip_images}")
    
    try:
        async with GCGScraper(lang=args.lang, max_connections=args.max_conn) as scraper:
            cards = await scraper.scrape_all_cards(download_images=not args.skip_images)
            scraper.save_outputs(cards)
            
        logger.info("Scraping completed successfully!")
        logger.info(f"Total cards processed: {len(cards)}")
        
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
