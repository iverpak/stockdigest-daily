# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server locally
uvicorn app:APP --host 0.0.0.0 --port 8000

# Run with auto-reload for development
uvicorn app:APP --reload --host 0.0.0.0 --port 8000
```

### PowerShell Automation Script
```powershell
# Execute the main automation workflow
.\scripts\setup.ps1
```

This PowerShell script orchestrates the complete QuantBrief workflow including initialization, feed cleaning, ingestion, and digest generation with configurable parameters for tickers, batch sizes, and time windows.

## Project Architecture

### Core Application Structure

**QuantBrief** is a financial news aggregation and analysis system built with FastAPI. The architecture consists of:

- **Single-file monolithic design**: All functionality is contained in `app.py` (~12,000+ lines)
- **PostgreSQL database**: Stores articles, ticker metadata, and processing state
- **AI-powered content analysis**: Uses OpenAI API for article summarization and relevance scoring
- **Multi-source content scraping**: Supports various scraping strategies including Playwright for anti-bot domains

### Key Components

#### Data Models and Storage
- Ticker reference data stored in PostgreSQL with CSV backup (`data/ticker_reference.csv`)
- Articles table with deduplication via URL hashing
- Metadata tracking for company information and processing state

#### Content Pipeline
1. **Feed Ingestion** (`/cron/ingest`): RSS feed parsing and article discovery
2. **Content Scraping**: Multi-strategy approach with fallbacks (newspaper3k → Playwright → ScrapingBee → ScrapFly)
3. **AI Triage**: OpenAI-powered relevance scoring and categorization
4. **Digest Generation** (`/cron/digest`): Email compilation using Jinja2 templates

#### Anti-Bot and Rate Limiting
- Domain-specific scraping strategies defined in `get_domain_strategy()`
- Playwright-based browser automation for challenging sites
- Built-in semaphore controls for concurrent processing
- User-agent rotation and referrer spoofing

### Memory Management

The `memory_monitor.py` module provides comprehensive resource tracking including:
- Database connection monitoring
- Playwright browser instance cleanup
- Async task lifecycle management
- Memory snapshot comparisons

### API Endpoints

#### Admin Endpoints (require X-Admin-Token header)
- `POST /admin/init`: Initialize ticker feeds and sync reference data
- `POST /admin/clean-feeds`: Clean old articles beyond time window
- `POST /admin/force-digest`: Generate digest emails for specific tickers
- `POST /admin/wipe-database`: Complete database reset
- `GET /admin/ticker-metadata/{ticker}`: Retrieve ticker configuration

#### Automation Endpoints
- `POST /cron/ingest`: RSS feed processing and article discovery
- `POST /cron/digest`: Email digest generation and delivery

### Configuration

#### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (required)
- `OPENAI_API_KEY`: OpenAI API access (required for AI features)
- `ADMIN_TOKEN`: Authentication for admin endpoints
- Email configuration: `SMTP_*` variables for digest delivery

#### Default Processing Parameters
- Time window: 1440 minutes (24 hours)
- Triage batch size: 2-3 articles per AI call
- Scraping concurrency: Controlled via semaphores
- Default ticker set: ["MO", "GM", "ODFL", "SO", "CVS"]

### Database Schema

Key tables managed through schema initialization:
- `articles`: Content storage with URL deduplication
- `ticker_references`: Company metadata and exchange information
- Processing state tracking for digest generation

### Content Processing Strategy

#### Article Categorization
Articles are automatically categorized into:
- **Company**: Direct company mentions
- **Sector**: Industry-related content
- **Competitor**: Competitive landscape analysis
- **Market**: Broader market context

#### Quality Scoring
Multi-tier domain quality assessment:
- Tier 1: Premium financial sources (WSJ, Bloomberg, Reuters)
- Tier 2: Established business media
- Tier 3: General news sources
- Tier 4: Lower-quality domains with content filtering

### Email Template System

Uses Jinja2 templating (`email_template.html`) with:
- Responsive design for email clients
- Categorized article sections
- Publisher attribution and timestamps
- Toronto timezone standardization (America/Toronto)

## Development Notes

- The application uses extensive logging for debugging and monitoring
- All ticker symbols are normalized and validated against exchange patterns
- Robust error handling with fallback strategies for content extraction
- Built-in rate limiting and respect for robots.txt files