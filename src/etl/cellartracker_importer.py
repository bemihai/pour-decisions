"""
CellarTracker API importer for wine cellar database.
"""
from typing import Dict, List, Optional
from datetime import datetime
from cellartracker import cellartracker

from src.database import Wine, Bottle
from src.database.repository import (
    SyncLogRepository, WineRepository, BottleRepository, ProducerRepository, RegionRepository, TastingRepository
)
from src.etl.utils import (
    normalize_wine_type,
    clean_text,
    parse_date,
    parse_vintage,
    parse_drinking_window, parse_country, parse_float, parse_int, parse_bool
)
from src.utils import get_default_db_path
from src.utils.logger import logger


class CellarTrackerImporter:
    """Import wine cellar-data from CellarTracker API."""

    def __init__(self, username: str, password: str, db_path: str = 'cellar-data/wine_cellar.db'):
        """
        Initialize CellarTracker importer.

        Args:
            username: CellarTracker username
            password: CellarTracker password
            db_path: Path to SQLite database
        """
        self.client = cellartracker.CellarTracker(username, password)
        self.db_path = db_path or get_default_db_path()
        self.stats = {
            'wines_processed': 0,
            'wines_imported': 0,
            'wines_updated': 0,
            'wines_skipped': 0,
            'bottles_processed': 0,
            'bottles_imported': 0,
            'bottles_updated': 0,
            'producers_created': 0,
            'regions_created': 0,
            'notes_processed': 0,
            'errors': []
        }
        self.sync_log_repo = SyncLogRepository(self.db_path)
        self.wine_repo = WineRepository(self.db_path)
        self.bottle_repo = BottleRepository(self.db_path)
        self.producer_repo = ProducerRepository(self.db_path)
        self.region_repo = RegionRepository(self.db_path)
        self.tasting_repo = TastingRepository(self.db_path)

    def import_all(self) -> Dict:
        """
        Import all cellar-data from CellarTracker following recommended strategy.

        Import Order:
        1. inventory.json - Build Wine catalog + current Bottle inventory
        2. bottles.json - Complete bottle lifecycle (adds drinking windows to Wines)
        3. notes.json - Enhance Wines with ratings & tasting notes

        Returns:
            Import statistics dictionary
        """
        logger.info("Starting full CellarTracker import")
        sync_id = self.sync_log_repo.start_sync_log("full")

        try:
            logger.info("Step 1/4: Fetching and importing inventory...")
            inventory = self.client.get_inventory()
            self._process_inventory(inventory)

            logger.info("Step 2/4: Fetching and importing availability cellar-data...")
            available = self.client.get_availability()
            self._process_availability(available)

            logger.info("Step 2/3: Fetching and importing bottles (complete history)...")
            bottles = self.client.get_bottles()
            self._process_bottles(bottles)

            logger.info("Step 3/3: Fetching and importing tasting notes...")
            notes = self.client.get_notes()
            self._process_tasting_notes(notes)

            self.sync_log_repo.complete_sync_log(sync_id, self.stats, status="success")
            logger.info(f"✅ Import completed successfully!")

        except Exception as e:
            error_msg = f"Import failed: {e}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            self.sync_log_repo.complete_sync_log(sync_id, self.stats, "failed", error_msg)

        return self.stats


    def _process_inventory(self, inventory: List[Dict]):
        """
        Process inventory - the current cellar snapshot.

        Creates:
        - Wine entities with catalog info (name, producer, vintage, type, etc.)
        - Bottle entities for current cellar (location, purchase info, status='in_cellar')
        """
        logger.info(f"Processing {len(inventory)} bottles from inventory")

        for record in inventory:
            try:
                self.stats["wines_processed"] += 1
                iwine = record.get("iWine")
                wine = self._get_wine_object_from_inventory_record(record)
                if existing := self.wine_repo.get_by_external_id(iwine):
                    wine.id = existing.id
                    wine_id = existing.id
                    self.wine_repo.update(wine)
                    self.stats["wines_updated"] += 1
                    logger.debug(f"Updated wine: {wine.wine_name} ({wine.vintage})")
                else:
                    wine_id = self.wine_repo.create(wine)
                    self.stats["wines_imported"] += 1
                    logger.debug(f"Imported wine: {wine.wine_name} ({wine.vintage})")

                # Upsert community rating from CT field (available for most inventory wines)
                ct_score = parse_float(record.get("CT"))
                if ct_score is not None:
                    self.tasting_repo.upsert_community_data(
                        wine_id=wine_id,
                        community_rating=ct_score,
                        like_votes=None,
                        like_percentage=None,
                    )

                bottle = self._get_bottle_object_from_inventory_record(record, wine_id)
                barcode = record.get("Barcode")
                if existing := self.bottle_repo.get_by_wine_and_external_id(wine_id, barcode):
                    bottle.id = existing.id
                    self.bottle_repo.update(bottle)
                    logger.debug(f"Updated bottle: {barcode}")
                else:
                    bottle.quantity = 1
                    bottle.status = "in_cellar"
                    self.bottle_repo.create(bottle)
                    logger.debug(f"Imported bottle: {barcode}")
            except Exception as e:
                error_msg = f"Error processing inventory record {record.get('iWine')}/{record.get('Barcode')}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)


    def _process_availability(self, available: List[Dict]):
        """
        Process availability cellar-data - Updates wine catalog with drinking index scores.

        Updates Wine entities with:
        - drink_index (availability score from Available column, converted to 0-100 scale)
        """
        logger.info(f"Processing {len(available)} wines from availability cellar-data")

        for record in available:
            try:
                iwine = record.get("iWine")
                wine = self.wine_repo.get_by_external_id(iwine)

                if not wine:
                    logger.debug(f"Wine {iwine} not found in availability processing, skipping")
                    continue

                drink_index = record.get("Available")
                if drink_index:
                    try:
                        if drink_index != wine.drink_index:
                            wine.drink_index = drink_index
                            self.wine_repo.update(wine)
                            logger.debug(f"Updated drink_index for wine {iwine}: {drink_index}")
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse available score '{drink_index}' for wine {iwine}")

            except Exception as e:
                error_msg = f"Error processing availability record for wine {record.get('iWine')}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)


    def _process_bottles(self, bottles: List[Dict]):
        """
        Process bottles - Complete bottle lifecycle.

        Creates/Updates:
        - Bottle: Complete lifecycle (in_cellar, consumed, gifted, lost)
        """
        logger.info(f"Processing {len(bottles)} bottles from complete lifecycle")

        for record in bottles:
            self.stats["bottles_processed"] += 1

            try:
                iwine = record.get("iWine")
                wine = self.wine_repo.get_by_external_id(iwine)

                if not wine:
                    wine = self._get_wine_object_from_inventory_record(record)
                    wine_id = self.wine_repo.create(wine)
                    self.stats["wines_processed"] += 1
                    self.stats["wines_imported"] += 1
                    logger.debug(f"Created wine from bottles: {wine.wine_name}")
                else:
                    wine_id = wine.id

                bottle = self._get_bottle_object_from_bottles_record(record, wine_id)
                barcode = record.get("Barcode")

                if existing := self.bottle_repo.get_by_wine_and_external_id(wine_id, barcode):
                    bottle.id = existing.id
                    self.bottle_repo.update(bottle)
                    self.stats["bottles_updated"] += 1
                    logger.debug(f"Updated bottle from bottles: {barcode}")
                else:
                    self.bottle_repo.create(bottle)
                    self.stats["bottles_imported"] += 1
                    logger.debug(f"Imported bottle from bottles: {barcode}")

            except Exception as e:
                error_msg = f"Error processing bottle {record.get('Barcode')}: {e}"
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)


    def _process_tasting_notes(self, notes: List[Dict]):
        """
        Process notes - Tasting notes and ratings.

        For every note record:
        - Community data (CScore, LikeVotes, LikePercent) is upserted unconditionally so
          wines without a personal review still have community ratings in the tastings table.
        - Personal data (rating, notes, do_like, is_defective) is merged into the same row.
        """
        logger.info(f"Processing {len(notes)} tasting notes")

        for record in notes:
            self.stats["notes_processed"] += 1
            try:
                iwine = record.get("iWine")
                if not iwine or not isinstance(iwine, (str, int)) or str(iwine).strip().lower() in ("true", "false", ""):
                    logger.warning(f"Skipping note with invalid iWine value: {iwine}")
                    continue

                wine = self.wine_repo.get_by_external_id(str(iwine))
                if not wine:
                    logger.warning(f"Wine {iwine} not found for note update")
                    continue

                wine_id = wine.id

                # Always upsert community data — this creates the tasting row if missing
                community_rating = parse_float(record.get("CScore"))
                like_votes = parse_int(record.get("LikeVotes"))
                like_percentage = parse_float(record.get("LikePercent"))

                community_kwargs = {"wine_id": wine_id}
                if community_rating is not None:
                    community_kwargs["community_rating"] = community_rating
                if like_votes is not None:
                    community_kwargs["like_votes"] = like_votes
                if like_percentage is not None:
                    community_kwargs["like_percentage"] = like_percentage

                self.tasting_repo.upsert_community_data(**community_kwargs)
                # Merge personal review data if present
                personal_rating = self._extract_rating_from_note(record)
                personal_notes = self._extract_tasting_notes_from_note(record, "")
                do_like = parse_bool(record.get("fLikeIt"))
                is_defective = parse_bool(record.get("Defective"))
                tasting_date_str = parse_date(record.get("TastingDate"))

                has_personal_data = any([personal_rating, personal_notes, do_like is not None, is_defective])
                if not has_personal_data:
                    continue

                existing = self.tasting_repo.get_latest_by_wine(wine_id)
                if not existing:
                    continue  # Should not happen after upsert above, but guard anyway

                updated = False

                if personal_rating and (not existing.personal_rating or personal_rating > existing.personal_rating):
                    existing.personal_rating = personal_rating
                    updated = True

                merged_notes = self._extract_tasting_notes_from_note(record, existing.tasting_notes or "")
                if merged_notes != existing.tasting_notes:
                    existing.tasting_notes = merged_notes
                    updated = True

                if tasting_date_str:
                    from datetime import date as date_cls
                    tasting_date = date_cls.fromisoformat(tasting_date_str)
                    if not existing.last_tasted_date or tasting_date > existing.last_tasted_date:
                        existing.last_tasted_date = tasting_date
                        updated = True

                if do_like is not None and existing.do_like is None:
                    existing.do_like = do_like
                    updated = True

                if is_defective and not existing.is_defective:
                    existing.is_defective = True
                    updated = True

                if updated:
                    self.tasting_repo.update(existing)
                    logger.debug(f"Updated personal tasting data for wine {iwine}")

            except Exception as e:
                error_msg = f"Error processing note {record.get('iNote')}: {e}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)


    def _get_wine_object_from_inventory_record(self, record: Dict) -> Wine:
        """
        Create a Wine object from an inventory record.
        """
        iwine = record.get("iWine")
        wine_name = clean_text(record.get("Wine", ""))
        vintage = parse_vintage(record.get("Vintage"))
        wine_type = normalize_wine_type(record.get("Type", ""))

        producer_id = self.producer_repo.get_or_create(
            clean_text(record.get("Producer", "")),
            parse_country(record.get("Country")),
            clean_text(record.get("Locale")),
        )

        region_primary = clean_text(record.get("Region"))
        region_secondary = clean_text(record.get("SubRegion")) or clean_text(record.get("Appellation"))
        region_id = self.region_repo.get_or_create(
            region_primary,
            parse_country(record.get("Country")),
            region_secondary,
        )

        q_purchased = int(record.get("PurchasedCommunity", 0) or 0)
        q_quantity = int(record.get("QuantityCommunity", 0) or 0)
        q_consumed = int(record.get("ConsumedCommunity", 0) or 0)

        drink_from_year, drink_to_year = parse_drinking_window(
            record.get("BeginConsume"),
            record.get("EndConsume")
        )

        return Wine(
            source="cellar_tracker",
            external_id=iwine,
            wine_name=wine_name,
            producer_id=producer_id,
            vintage=vintage,
            wine_type=wine_type,
            varietal=clean_text(record.get("Varietal")),
            designation=clean_text(record.get("Designation")),
            region_id=region_id,
            appellation=clean_text(record.get("Appellation")),
            vineyard=clean_text(record.get("Vineyard")),
            bottle_size=record.get("Size", "750ml"),
            drink_from_year=drink_from_year,
            drink_to_year=drink_to_year,
            q_purchased=q_purchased,
            q_quantity=q_quantity,
            q_consumed=q_consumed
        )

    @staticmethod
    def _get_bottle_object_from_inventory_record(record: Dict, wine_id: int) -> Bottle:
        """
        Create a Bottle object from an inventory record.

        Args:
            record: Inventory CSV record
            wine_id: Wine ID
        """
        purchase_price = None
        price_str = record.get("Price")
        if price_str:
            try:
                purchase_price = float(price_str)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse price '{price_str}' in record: {record.get('Barcode')}")

        valuation_price = None
        valuation_str = record.get("Valuation")
        if valuation_str:
            try:
                valuation_price = float(valuation_str)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse valuation '{valuation_str}' in record: {record.get('Barcode')}")

        return Bottle(
            wine_id=wine_id,
            source="cellar_tracker",
            external_bottle_id=record.get("Barcode"),
            location=clean_text(record.get("Location")),
            bin=clean_text(record.get("Bin")),
            purchase_date=parse_date(record.get("PurchaseDate")),
            bottle_note=clean_text(record.get("BottleNote")),
            purchase_price=purchase_price,
            valuation_price=valuation_price,
            currency=record.get("Currency", "RON"),
            store_name=clean_text(record.get("StoreName"))
        )


    def _get_bottle_object_from_bottles_record(self, record: Dict, wine_id: int) -> Bottle:
        """
        Create or update Bottle from bottles.csv record.

        Args:
            record: Bottles CSV record
            wine_id: Wine ID
        """
        barcode = record.get("Barcode")
        quantity = int(record.get("Quantity", 1))
        bottle_state = record.get("BottleState", "1")
        consumption_date = record.get("ConsumptionDate")

        if bottle_state == "1" or (bottle_state == "0" and not consumption_date):
            status = "in_cellar"
        elif consumption_date:
            short_type = record.get("ShortType", "").lower()
            if "gift" in short_type:
                status = "gifted"
            elif "spoil" in short_type or "dump" in short_type:
                status = "lost"
            else:
                status = "consumed"
        else:
            status = "in_cellar"

        purchase_date = parse_date(record.get("PurchaseDate"))
        consumed_date = parse_date(consumption_date) if consumption_date else None

        purchase_price = None
        price_str = record.get("BottleCost")
        if price_str:
            try:
                purchase_price = float(price_str)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse bottle cost '{price_str}' for {barcode}")

        purchase_note = clean_text(record.get("PurchaseNote"))
        consumption_note = clean_text(record.get("ConsumptionNote"))
        bottle_note = self._merge_bottle_notes(purchase_note, consumption_note)

        return Bottle(
            wine_id=wine_id,
            source="cellar_tracker",
            external_bottle_id=barcode,
            quantity=quantity,
            status=status,
            location=clean_text(record.get("Location")),
            bin=clean_text(record.get("Bin")),
            purchase_date=purchase_date,
            purchase_price=purchase_price,
            currency=record.get("BottleCostCurrency", "RON"),
            store_name=clean_text(record.get("Store")),
            consumed_date=consumed_date,
            bottle_note=bottle_note
        )


    @staticmethod
    def _merge_bottle_notes(purchase_note: Optional[str], consumption_note: Optional[str]) -> Optional[str]:
        """Merge purchase and consumption notes."""
        if purchase_note and consumption_note:
            return f"{purchase_note}\n\nConsumed: {consumption_note}"
        return purchase_note or consumption_note

    @staticmethod
    def _extract_rating_from_note(record: Dict) -> Optional[int]:
        """Extract personal rating from note record (0-100 scale)."""
        rating_str = record.get("Rating")
        if rating_str:
            try:
                return int(rating_str)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert rating '{rating_str}' to int: {e}")
        return None

    @staticmethod
    def _extract_tasting_notes_from_note(record: Dict, existing_notes: str) -> str:
        """Extract and merge tasting notes from note record with date stamps."""
        tasting_date = parse_date(record.get("TastingDate"))
        note_text = clean_text(record.get("TastingNotes"))

        if note_text:
            date_str = tasting_date if tasting_date else datetime.now().strftime('%Y-%m-%d')
            new_note_entry = f"[{date_str}] {note_text}"

            # Check if this exact note already exists in existing_notes
            if existing_notes and new_note_entry in existing_notes:
                # Note already exists, don't duplicate
                return existing_notes

            if existing_notes:
                return f"{existing_notes}\n\n{new_note_entry}"
            else:
                return new_note_entry

        return existing_notes


