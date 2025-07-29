# Gundam GCG Card Scraper

A comprehensive, well-documented scraper for the official Gundam GCG (Gundam Card Game) website. Downloads all card metadata, artwork, and set information from [https://www.gundam-gcg.com](https://www.gundam-gcg.com).

## Features

-   **Complete Coverage**: Scrapes all card sets (base, parallels, promos, future releases)
-   **High-Resolution Artwork**: Downloads full-quality card images
-   **Structured Data**: Extracts comprehensive metadata (stats, text, keywords, etc.)
-   **Multi-Language Support**: Supports multiple languages (en, jp, zh-tw, etc.)
-   **Parallel Processing**: Efficient concurrent downloads with configurable limits
-   **Resumable Downloads**: Skips existing images for safe re-runs
-   **Multiple Output Formats**: JSON and CSV outputs
-   **Robust Error Handling**: Graceful handling of network issues and parsing errors
-   **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Requirements

-   Python >= 3.9
-   aiohttp
-   beautifulsoup4
-   lxml
-   tqdm

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd gundam-gcg-scraper
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Scrape English cards with images (default)
python scrape_gcg.py

# Scrape Japanese cards
python scrape_gcg.py --lang jp

# Metadata only, skip image downloads
python scrape_gcg.py --skip-images

# Use fewer connections (be more polite to the server)
python scrape_gcg.py --max-conn 8

# Enable verbose logging
python scrape_gcg.py --verbose
```

### Command Line Options

| Option          | Description                              | Default |
| --------------- | ---------------------------------------- | ------- |
| `--lang`        | Language to scrape (en, jp, zh-tw, etc.) | `en`    |
| `--max-conn`    | Maximum parallel connections             | `16`    |
| `--skip-images` | Skip artwork downloads                   | `False` |
| `--verbose`     | Enable verbose logging                   | `False` |

### Output Files

The scraper generates the following output files:

-   **`gundam_cards.json`** - Complete card data with Unicode preserved
-   **`gundam_cards.csv`** - Excel-friendly flat file
-   **`card_images/`** - High-resolution artwork (one image per card)

## Code Structure

The scraper is organized into a clean, object-oriented structure:

### Core Classes

-   **`GCGScraper`**: Main scraper class with async context manager support
-   **`SetInfo`**: Data class representing card set information
-   **`CardData`**: Data class representing structured card data

### Key Methods

-   **`discover_sets()`**: Discovers all available card sets
-   **`gather_card_urls()`**: Collects all card detail URLs from a set
-   **`parse_card_detail()`**: Parses individual card pages
-   **`scrape_all_cards()`**: Orchestrates the entire scraping process

### Error Handling

The scraper includes comprehensive error handling:

-   Network timeouts and connection errors
-   HTML parsing failures
-   Image download failures
-   Graceful degradation for missing data

## Data Schema

### Card Data Structure

Each card contains the following fields:

```json
{
    "card_id": "string",
    "card_number": "string",
    "name": "string",
    "level": "string",
    "cost": "string",
    "color": "string",
    "type": "string",
    "ap": "string",
    "hp": "string",
    "text": "string",
    "zone": "string",
    "trait": "string",
    "link": "string",
    "rarity": "string",
    "source": "string",
    "set": "string",
    "where_to_get_it": "string",
    "image": "string",
    "pilot": "string",
    "keywords": ["string"]
}
```

### Keywords and Tags

The scraper automatically extracts:

-   **Action Types**: From square brackets `[Action]`
-   **Keywords**: From angle brackets `<Keyword>`
-   **Pilot Names**: From `[Pilot][Name]` patterns

## Performance Considerations

-   **Rate Limiting**: Default 16 concurrent connections (configurable)
-   **Resumable**: Skips existing images for efficient re-runs
-   **Memory Efficient**: Processes cards in batches
-   **Network Resilient**: Handles timeouts and connection errors

## Troubleshooting

### Common Issues

1. **Connection Errors**: Reduce `--max-conn` value
2. **Timeout Errors**: Check network connectivity
3. **Parsing Errors**: Website structure may have changed
4. **Memory Issues**: Process smaller batches or skip images

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python scrape_gcg.py --verbose
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   Gundam GCG website for providing the card data
-   BeautifulSoup and aiohttp communities for excellent libraries
-   Open source contributors who maintain the dependencies
