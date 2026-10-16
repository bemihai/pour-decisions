"""Vivino CSV importer for wine cellar database."""
import csv
from pathlib import Path

from src.database.models import Wine, Bottle, Tasting
from src.database.repository import (
    ProducerRepository, RegionRepository, WineRepository, BottleRepository, TastingRepository
)
from src.etl.utils import (
    normalize_wine_type,
    clean_text,
    convert_vivino_community_rating,
    convert_vivino_personal_rating,
    parse_date,
    parse_vintage,
    parse_drinking_window,
    parse_vivino_rating,
    generate_external_id,
    parse_country
)
from src.utils import get_default_db_path, logger


def load_full_wine_list_data(csv_path: str | Path) -> dict[tuple[str, str, int | None], dict]:
    """Load and group a Vivino full-wine-list export using importer rules.

    Args:
        csv_path: Path to the Vivino CSV export.

    Returns:
        Mapping from cleaned winery, wine name, and vintage to grouped scans.
    """
    wines_data: dict[tuple[str, str, int | None], dict] = {}
    with open(csv_path, "r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            winery = clean_text(row.get("Winery"))
            wine_name = clean_text(row.get("Wine name"))
            vintage = parse_vintage(row.get("Vintage"))
            if not (winery and wine_name):
                continue

            key = (winery, wine_name, vintage)
            if key not in wines_data:
                wines_data[key] = {
                    "row": row,
                    "scans": [],
                    "ratings": [],
                    "invalid_ratings": 0,
                    "reviews": [],
                }

            data = wines_data[key]
            data["scans"].append(row)

            raw_rating = row.get("Your rating")
            if raw_rating not in (None, ""):
                parsed_rating = parse_vivino_rating(raw_rating)
                if parsed_rating is None:
                    data["invalid_ratings"] += 1
                else:
                    data["ratings"].append(parsed_rating)

            review = clean_text(row.get("Your review") or row.get("Personal Note"))
            if review:
                data["reviews"].append(review)

    return wines_data


class VivinoImporter:
    """Import wine cellar-data from Vivino CSV exports."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize Vivino importer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or get_default_db_path()
        self.producer_repository = ProducerRepository(self.db_path)
        self.region_repository = RegionRepository(self.db_path)
        self.wine_repository = WineRepository(self.db_path)
        self.bottle_repository = BottleRepository(self.db_path)
        self.tasting_repository = TastingRepository(self.db_path)

        self.stats = {
            'wines_processed': 0,
            'wines_imported': 0,
            'wines_updated': 0,
            'wines_skipped': 0,
            'bottles_imported': 0,
            'bottles_updated': 0,
            'producers_created': 0,
            'regions_created': 0,
            'errors': []
        }

    def import_full_wine_list_csv(self, csv_path: str | Path) -> dict:
        """
        Import wines from Vivino full_wine_list.csv export.
        This includes tasting notes and ratings.

        Args:
            csv_path: Path to full_wine_list.csv file

        Returns:
            Import statistics dictionary
        """
        logger.info(f"Starting import from {csv_path}")

        try:
            wines_data = load_full_wine_list_data(csv_path)

            # Process each unique wine
            for key, data in wines_data.items():
                self.stats["wines_processed"] += 1

                try:
                    self._process_full_wine_list_data(data)
                except Exception as e:
                    error_msg = f"Error processing wine {key}: {e}"
                    logger.error(error_msg)
                    self.stats["errors"].append(error_msg)

            logger.info("Import completed.")
            return self.stats

        except Exception as e:
            logger.error(f"Failed to import from {csv_path}: {e}")
            self.stats['errors'].append(str(e))
            return self.stats

    def _process_full_wine_list_data(self, data: dict):
        """Process aggregated cellar-data from full_wine_list.csv."""

        if not (wine_import := self._create_wine_object_from_data(data)):
            self.stats["wines_skipped"] += 1
            return

        # Check if wine already exists: if yes, update; if no, insert
        if wine_record := self.wine_repository.get_by_external_id(wine_import.external_id):
            wine_import.id = wine_record.id
            wine_id = wine_record.id
            self.wine_repository.update(wine_import)
            self.stats["wines_updated"] += 1
        elif duplicates := self.wine_repository.find_duplicates(
            wine_import.wine_name,
            data["row"]["Winery"],
            wine_import.wine_type,
            wine_import.vintage,
        ):
            self.stats["wines_skipped"] += 1
            logger.debug(f"Found duplicate wines for {wine_import.wine_name} ({wine_import.vintage}): {duplicates}")
            return
        else:
            wine_id = self.wine_repository.create(wine_import)
            self.stats['wines_imported'] += 1

        # Create or update tasting record
        tasting_id = self._create_or_update_tasting(data, wine_id)

        # Create bottle record (quantity = 1 for full_wine_list as it tracks activity not inventory)
        bottle = Bottle(
            wine_id=wine_id,
            tasting_id=tasting_id,
            source="vivino",
            external_bottle_id=wine_import.external_id,
            quantity=1,
            status="consumed",
            consumed_date=self._get_last_tasted_date(data) or parse_date("2010-01-01"),
        )

        if existing_bottle := self.bottle_repository.get_by_wine_and_external_id(wine_id, bottle.external_bottle_id):
            bottle.id = existing_bottle.id
            self.bottle_repository.update(bottle)
            self.stats["bottles_updated"] += 1
        else:
            self.bottle_repository.create(bottle)
            self.stats["bottles_imported"] += 1

    def _create_wine_object_from_data(self, data: dict) -> Wine | None:
        """Create Wine object from aggregated cellar-data (without tasting fields)."""
        row = data["row"]
        winery = clean_text(row["Winery"])
        wine_name = clean_text(row["Wine name"])
        country = parse_country(row["Country"])

        # Validate required fields and return None if missing
        if not (winery and wine_name and country):
            return None

        # Prepare fields for insertion/update
        vintage = parse_vintage(row.get("Vintage"))
        region = clean_text(row["Region"])
        wine_type = normalize_wine_type(row["Wine type"])
        drink_from, drink_to = parse_drinking_window(row.get("Drinking Window", ""))
        external_id = generate_external_id(winery, wine_name, vintage)

        # Create producer and track if new
        existing_producer = self.producer_repository.get_by_name(winery)
        producer_id = self.producer_repository.get_or_create(winery, country, region)
        if not existing_producer:
            self.stats['producers_created'] += 1

        # Split region into primary and secondary if it contains a separator
        primary_region = region
        secondary_region = None
        if region and ' - ' in region:
            parts = region.split(' - ', 1)
            primary_region = parts[0].strip()
            secondary_region = parts[1].strip() if len(parts) > 1 else None

        # Create region and track if new
        existing_region = self.region_repository.get_by_name_and_country(primary_region, country, secondary_region)
        region_id = self.region_repository.get_or_create(primary_region, country, secondary_region)
        if not existing_region:
            self.stats['regions_created'] += 1

        return Wine(
            source="vivino",
            wine_name=wine_name,
            wine_type=wine_type,
            vintage=vintage,
            drink_from_year=drink_from,
            drink_to_year=drink_to,
            external_id=external_id,
            producer_id=producer_id,
            region_id=region_id,
        )

    def _create_or_update_tasting(self, data: dict, wine_id: int) -> int | None:
        """
        Create or update tasting record from Vivino cellar-data.

        Args:
            data: Aggregated wine cellar-data with ratings and reviews
            wine_id: Wine ID to associate tasting with

        Returns:
            Tasting ID if created/updated, None otherwise
        """
        row = data["row"]

        # Select the highest valid personal rating across all scans.
        raw_personal_rating = max(data["ratings"]) if data["ratings"] else None
        personal_rating = convert_vivino_personal_rating(raw_personal_rating)
        do_like = raw_personal_rating >= 3.5 if raw_personal_rating is not None else None

        # Get last tasted date
        last_tasted_str = self._get_last_tasted_date(data)
        last_tasted = None
        if last_tasted_str:
            from datetime import date as date_cls
            try:
                last_tasted = date_cls.fromisoformat(last_tasted_str)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse date: {last_tasted_str}")
                last_tasted = None

        # Merge tasting notes with date prefixes
        tasting_notes = None
        if data["reviews"]:
            # Prefix each review with its scan date
            dated_notes = []
            for scan in data["scans"]:
                review = clean_text(scan.get("Your review") or scan.get("Personal Note"))
                if review:
                    scan_date = parse_date(scan.get("Scan date"))
                    date_str = scan_date if scan_date else "Unknown date"
                    dated_notes.append(f"[{date_str}] {review}")

            # Remove duplicates while preserving order
            seen = set()
            unique_notes = []
            for note in dated_notes:
                if note not in seen:
                    seen.add(note)
                    unique_notes.append(note)

            tasting_notes = '\n\n'.join(unique_notes) if unique_notes else None

        # Extract community rating (convert from 0-5 to 0-100 scale if enough votes)
        community_rating = None
        if row.get("Average rating"):
            try:
                avg_rating = float(row["Average rating"])
                # Only use community rating if there are enough votes (placeholder logic)
                community_rating = convert_vivino_community_rating(avg_rating)
            except (ValueError, TypeError):
                pass

        # Check if we have any tasting cellar-data
        if personal_rating is None and tasting_notes is None and community_rating is None:
            return None

        # Get existing tasting or create new one
        existing_tasting = self.tasting_repository.get_latest_by_wine(wine_id)

        if existing_tasting:
            # Update existing tasting
            updated = False
            personal_fields_changed = False

            # A Vivino re-import is authoritative even when recalibration lowers the score.
            if personal_rating is not None and personal_rating != existing_tasting.personal_rating:
                existing_tasting.personal_rating = personal_rating
                updated = True
                personal_fields_changed = True
            if do_like is not None and do_like != existing_tasting.do_like:
                existing_tasting.do_like = do_like
                updated = True
                personal_fields_changed = True

            # Merge tasting notes
            if tasting_notes:
                if existing_tasting.tasting_notes:
                    # Check if the new notes are already present to avoid duplication
                    new_notes_list = tasting_notes.split('\n\n')
                    existing_notes_list = existing_tasting.tasting_notes.split('\n\n')

                    # Add only notes that don't already exist
                    notes_to_add = [note for note in new_notes_list if note not in existing_notes_list]

                    if notes_to_add:
                        combined_notes = existing_tasting.tasting_notes + '\n\n' + '\n\n'.join(notes_to_add)
                        existing_tasting.tasting_notes = combined_notes
                        updated = True
                else:
                    existing_tasting.tasting_notes = tasting_notes
                    updated = True

            # Update community rating (always use latest)
            if community_rating is not None and community_rating != existing_tasting.community_rating:
                existing_tasting.community_rating = community_rating
                updated = True

            # Update tasting date (keep most recent)
            if last_tasted and (not existing_tasting.last_tasted_date or last_tasted > existing_tasting.last_tasted_date):
                existing_tasting.last_tasted_date = last_tasted
                updated = True

            if updated:
                self.tasting_repository.update(existing_tasting)
                if personal_fields_changed:
                    self.tasting_repository.update_vivino_personal_rating(
                        existing_tasting.id,
                        personal_rating,
                        do_like,
                    )
                logger.debug(f"Updated tasting for wine {wine_id}")

            return existing_tasting.id
        else:
            # Create new tasting
            tasting = Tasting(
                wine_id=wine_id,
                personal_rating=personal_rating,
                tasting_notes=tasting_notes,
                community_rating=community_rating,
                last_tasted_date=last_tasted,
                do_like=do_like,
                is_defective=False
            )
            tasting_id = self.tasting_repository.create(tasting)
            logger.debug(f"Created tasting for wine {wine_id}")
            return tasting_id

    def _get_last_tasted_date(self, data: dict):
        """Extract the most recent tasting date from scan dates."""
        scan_dates = [parse_date(scan.get("Scan date")) for scan in data["scans"] if scan.get("Scan date")]
        return max(scan_dates) if scan_dates else None




