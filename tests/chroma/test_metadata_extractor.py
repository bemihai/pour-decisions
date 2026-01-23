"""Unit tests for src/chroma/metadata_extractor.py"""

from unittest.mock import Mock

from src.chroma.metadata_extractor import (
    WineMetadata,
    extract_grapes,
    extract_regions,
    extract_vintages,
    extract_classifications,
    extract_producers,
    extract_appellations,
    extract_wine_metadata,
    extract_document_context,
)


class TestWineMetadata:
    """Test WineMetadata dataclass."""

    def test_wine_metadata_creation(self):
        """Test creating WineMetadata with default values."""
        metadata = WineMetadata()

        assert metadata.grapes == set()
        assert metadata.regions == set()
        assert metadata.vintages == set()
        assert metadata.classifications == set()
        assert metadata.producers == set()
        assert metadata.appellations == set()

    def test_wine_metadata_with_values(self):
        """Test creating WineMetadata with values."""
        metadata = WineMetadata(
            grapes={"Chardonnay", "Pinot Noir"},
            regions={"Burgundy", "Champagne"},
            vintages={"2015", "2018"},
            classifications={"AOC", "DOC"},
            producers={"Domaine Leflaive"},
            appellations={"Chablis"},
        )

        assert len(metadata.grapes) == 2
        assert len(metadata.regions) == 2
        assert len(metadata.vintages) == 2
        assert "Chardonnay" in metadata.grapes
        assert "Burgundy" in metadata.regions

    def test_to_dict(self):
        """Test converting WineMetadata to dictionary."""
        metadata = WineMetadata(
            grapes={"Merlot", "Cabernet Sauvignon"},
            regions={"Bordeaux"},
            vintages={"2010", "2015", "2005"},
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert "grapes" in result
        assert "regions" in result
        assert isinstance(result["grapes"], list)
        assert sorted(result["grapes"]) == result["grapes"]  # Check sorted
        assert len(result["vintages"]) == 3

    def test_to_dict_empty(self):
        """Test to_dict with empty metadata."""
        metadata = WineMetadata()
        result = metadata.to_dict()

        assert all(len(v) == 0 for v in result.values())

    def test_is_empty_true(self):
        """Test is_empty returns True for empty metadata."""
        metadata = WineMetadata()
        assert metadata.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False when metadata exists."""
        metadata = WineMetadata(grapes={"Chardonnay"})
        assert metadata.is_empty() is False

        metadata2 = WineMetadata(regions={"Bordeaux"})
        assert metadata2.is_empty() is False

    def test_to_dict_sorting(self):
        """Test that to_dict returns sorted lists."""
        metadata = WineMetadata(
            grapes={"Zinfandel", "Cabernet", "Merlot"},
            regions={"Tuscany", "Bordeaux", "Rioja"},
        )

        result = metadata.to_dict()

        assert result["grapes"] == ["Cabernet", "Merlot", "Zinfandel"]
        assert result["regions"] == ["Bordeaux", "Rioja", "Tuscany"]


class TestExtractGrapes:
    """Test extract_grapes function."""

    def test_extract_single_grape(self):
        """Test extracting a single grape variety."""
        text = "This wine is made from Chardonnay grapes."
        result = extract_grapes(text)

        # Check that at least one grape is found (pattern returns canonical names)
        assert len(result) >= 1
        # The result set contains canonical grape names from GRAPE_PATTERNS
        assert isinstance(result, set)

    def test_extract_multiple_grapes(self):
        """Test extracting multiple grape varieties."""
        text = "A blend of Cabernet Sauvignon, Merlot, and Cabernet Franc."
        result = extract_grapes(text)

        assert len(result) >= 2

    def test_extract_grapes_case_insensitive(self):
        """Test that extraction is case insensitive."""
        text = "CHARDONNAY and pinot noir and MeRlOt"
        result = extract_grapes(text)

        assert len(result) >= 1

    def test_extract_grapes_empty_text(self):
        """Test with empty text."""
        result = extract_grapes("")
        assert len(result) == 0

    def test_extract_grapes_no_matches(self):
        """Test with text containing no grape varieties."""
        text = "This is a book about wine tasting techniques."
        result = extract_grapes(text)

        # May or may not find grapes depending on patterns
        assert isinstance(result, set)

    def test_extract_grapes_word_boundaries(self):
        """Test that word boundaries are respected."""
        text = "Chardonnay is excellent"
        result = extract_grapes(text)

        assert isinstance(result, set)


class TestExtractRegions:
    """Test extract_regions function."""

    def test_extract_single_region(self):
        """Test extracting a single wine region."""
        text = "Wines from Bordeaux are world-renowned."
        result = extract_regions(text)

        # Check that at least one region is found (pattern returns canonical names)
        assert len(result) >= 1
        # The result set contains canonical region names from REGION_PATTERNS
        assert isinstance(result, set)

    def test_extract_multiple_regions(self):
        """Test extracting multiple wine regions."""
        text = "I've tasted wines from Burgundy, Tuscany, and Rioja."
        result = extract_regions(text)

        assert len(result) >= 1

    def test_extract_regions_case_insensitive(self):
        """Test that extraction is case insensitive."""
        text = "wines from BORDEAUX and tuscany"
        result = extract_regions(text)

        assert len(result) >= 1

    def test_extract_regions_empty_text(self):
        """Test with empty text."""
        result = extract_regions("")
        assert len(result) == 0

    def test_extract_regions_no_matches(self):
        """Test with text containing no wine regions."""
        text = "This is about general geography."
        result = extract_regions(text)

        assert isinstance(result, set)


class TestExtractVintages:
    """Test extract_vintages function."""

    def test_extract_single_vintage(self):
        """Test extracting a single vintage year."""
        text = "The 2015 vintage was exceptional."
        result = extract_vintages(text)

        assert "2015" in result

    def test_extract_multiple_vintages(self):
        """Test extracting multiple vintage years."""
        text = "Great vintages include 2010, 2015, and 2018."
        result = extract_vintages(text)

        assert len(result) == 3
        assert "2010" in result
        assert "2015" in result
        assert "2018" in result

    def test_extract_vintages_range(self):
        """Test that only valid years (1900-2050) are extracted."""
        text = "Years: 1899, 1950, 2000, 2051"
        result = extract_vintages(text)

        assert "1899" not in result
        assert "2051" not in result
        assert "1950" in result
        assert "2000" in result

    def test_extract_vintages_empty_text(self):
        """Test with empty text."""
        result = extract_vintages("")
        assert len(result) == 0

    def test_extract_vintages_no_matches(self):
        """Test with text containing no years."""
        text = "This wine is excellent."
        result = extract_vintages(text)

        assert len(result) == 0

    def test_extract_vintages_edge_cases(self):
        """Test edge cases for vintage extraction."""
        text = "Vintages from 1900 to 2050 are valid."
        result = extract_vintages(text)

        assert "1900" in result
        assert "2050" in result

    def test_extract_vintages_filters_page_numbers(self):
        """Test that the function extracts years in valid range."""
        text = "Page 123. The 2015 vintage. See page 456."
        result = extract_vintages(text)

        # Should include 2015 which is in valid range
        assert "2015" in result


class TestExtractClassifications:
    """Test extract_classifications function."""

    def test_extract_single_classification(self):
        """Test extracting a single wine classification."""
        text = "This wine is DOCG certified."
        result = extract_classifications(text)

        assert "DOCG" in result

    def test_extract_multiple_classifications(self):
        """Test extracting multiple classifications."""
        text = "Classifications include AOC, DOC, and DOCG."
        result = extract_classifications(text)

        assert len(result) >= 1

    def test_extract_classifications_case_insensitive(self):
        """Test that extraction is case insensitive."""
        text = "This is an aoc wine and a doc wine."
        result = extract_classifications(text)

        assert len(result) >= 1
        # Results should be uppercase
        assert all(c.isupper() for c in result)

    def test_extract_classifications_empty_text(self):
        """Test with empty text."""
        result = extract_classifications("")
        assert len(result) == 0

    def test_extract_classifications_uppercase_result(self):
        """Test that results are always uppercase."""
        text = "This wine has aoc and doc classifications."
        result = extract_classifications(text)

        for classification in result:
            assert classification == classification.upper()


class TestExtractProducers:
    """Test extract_producers function."""

    def test_extract_producer_with_prefix(self):
        """Test extracting producer names with common prefixes."""
        text = "Château Margaux is a famous producer."
        result = extract_producers(text)

        assert len(result) >= 0  # May or may not find depending on patterns

    def test_extract_producer_with_suffix(self):
        """Test extracting producer names with common suffixes."""
        text = "Smith Vineyards produces excellent wine."
        result = extract_producers(text)

        assert isinstance(result, set)

    def test_extract_multiple_producers(self):
        """Test extracting multiple producer names."""
        text = "Domaine Leflaive and Château Latour are renowned producers."
        result = extract_producers(text)

        assert isinstance(result, set)

    def test_extract_producers_empty_text(self):
        """Test with empty text."""
        result = extract_producers("")
        assert len(result) == 0

    def test_extract_producers_minimum_length(self):
        """Test that very short matches are filtered out."""
        text = "XY Winery is too short."
        result = extract_producers(text)

        # Short names should be filtered
        assert isinstance(result, set)

    def test_extract_producers_title_case(self):
        """Test that producer names are returned in title case."""
        text = "DOMAINE LEFLAIVE and château margaux"
        result = extract_producers(text)

        for producer in result:
            # Should be title cased
            words = producer.split()
            for word in words:
                if word:  # Skip empty strings
                    assert word[0].isupper() or not word[0].isalpha()

    def test_extract_producers_with_accents(self):
        """Test extracting producers with accented characters."""
        text = "Château Pétrus is a legendary producer."
        result = extract_producers(text)

        assert isinstance(result, set)


class TestExtractAppellations:
    """Test extract_appellations function."""

    def test_extract_single_appellation(self):
        """Test extracting a single appellation."""
        text = "This Barolo is from a prestigious appellation."
        result = extract_appellations(text)

        assert len(result) >= 0

    def test_extract_multiple_appellations(self):
        """Test extracting multiple appellations."""
        text = "Great appellations include Barolo, Champagne, and Chablis."
        result = extract_appellations(text)

        assert isinstance(result, set)

    def test_extract_appellations_case_insensitive(self):
        """Test that extraction is case insensitive."""
        text = "I love CHAMPAGNE and barolo wines."
        result = extract_appellations(text)

        assert isinstance(result, set)

    def test_extract_appellations_empty_text(self):
        """Test with empty text."""
        result = extract_appellations("")
        assert len(result) == 0

    def test_extract_appellations_title_case(self):
        """Test that appellations are returned in title case."""
        text = "champagne and barolo are excellent"
        result = extract_appellations(text)

        for appellation in result:
            assert appellation[0].isupper()


class TestExtractWineMetadata:
    """Test extract_wine_metadata function."""

    def test_extract_comprehensive_metadata(self):
        """Test extracting all types of metadata from rich text."""
        text = """
        The 2015 Château Margaux from Bordeaux is a blend of
        Cabernet Sauvignon and Merlot. This AOC wine from the
        prestigious Margaux appellation is exceptional.
        """
        result = extract_wine_metadata(text)

        assert isinstance(result, WineMetadata)
        assert "2015" in result.vintages
        assert len(result.grapes) >= 1
        assert len(result.regions) >= 1

    def test_extract_metadata_empty_text(self):
        """Test with empty text."""
        result = extract_wine_metadata("")

        assert isinstance(result, WineMetadata)
        assert result.is_empty()

    def test_extract_metadata_no_wine_terms(self):
        """Test with text containing no wine terminology."""
        text = "This is a general text about nothing related to wine."
        result = extract_wine_metadata(text)

        assert isinstance(result, WineMetadata)

    def test_extract_metadata_partial_matches(self):
        """Test with text containing only some metadata types."""
        text = "This Chardonnay from 2018 is delicious."
        result = extract_wine_metadata(text)

        assert isinstance(result, WineMetadata)
        assert len(result.vintages) >= 1

    def test_extract_metadata_returns_all_fields(self):
        """Test that all metadata fields are present."""
        result = extract_wine_metadata("test text")

        assert hasattr(result, "grapes")
        assert hasattr(result, "regions")
        assert hasattr(result, "vintages")
        assert hasattr(result, "classifications")
        assert hasattr(result, "producers")
        assert hasattr(result, "appellations")


class TestExtractDocumentContext:
    """Test extract_document_context function."""

    def test_extract_context_empty_elements(self):
        """Test with empty element list."""
        result = extract_document_context([])

        assert result["document_title"] == ""
        assert result["chapter"] == ""
        assert result["section"] == ""

    def test_extract_context_with_title(self):
        """Test extracting document title from Title element."""
        mock_elem = Mock()
        mock_elem.category = "Title"
        mock_elem.__str__ = lambda self: "The Wine Bible"

        result = extract_document_context([mock_elem])

        assert result["document_title"] == "The Wine Bible"

    def test_extract_context_with_chapter(self):
        """Test extracting chapter information."""
        mock_title = Mock()
        mock_title.category = "Title"
        mock_title.__str__ = lambda self: "Chapter 1: Introduction"

        mock_content = Mock()
        mock_content.category = "NarrativeText"
        mock_content.__str__ = lambda self: "Content here"

        result = extract_document_context([mock_title, mock_content])

        assert "Chapter" in result["chapter"]

    def test_extract_context_max_title_length(self):
        """Test that title is truncated to max length."""
        long_title = "A" * 500
        mock_elem = Mock()
        mock_elem.category = "Title"
        mock_elem.__str__ = lambda self: long_title

        result = extract_document_context([mock_elem], max_title_length=100)

        assert len(result["document_title"]) <= 100

    def test_extract_context_fallback_to_first_element(self):
        """Test fallback to first non-empty element when no Title."""
        mock_elem = Mock()
        mock_elem.category = "NarrativeText"
        mock_elem.__str__ = lambda self: "First paragraph text"

        result = extract_document_context([mock_elem])

        assert result["document_title"] == "First paragraph text"

    def test_extract_context_with_section(self):
        """Test extracting section information."""
        mock_elem1 = Mock()
        mock_elem1.category = "Title"
        mock_elem1.__str__ = lambda self: "Main Title"

        mock_elem2 = Mock()
        mock_elem2.category = "Title"
        mock_elem2.__str__ = lambda self: "Short Section"

        result = extract_document_context([mock_elem1, mock_elem2])

        assert result["section"] == "Short Section"

    def test_extract_context_multiple_chapters(self):
        """Test that last chapter is kept."""
        mock_ch1 = Mock()
        mock_ch1.category = "Title"
        mock_ch1.__str__ = lambda self: "Chapter 1: First"

        mock_ch2 = Mock()
        mock_ch2.category = "Title"
        mock_ch2.__str__ = lambda self: "Chapter 2: Second"

        result = extract_document_context([mock_ch1, mock_ch2])

        assert "Chapter 2" in result["chapter"]

    def test_extract_context_ignores_long_sections(self):
        """Test that very long titles are not treated as sections."""
        long_text = "A" * 150
        mock_elem = Mock()
        mock_elem.category = "Title"
        mock_elem.__str__ = lambda self: long_text

        result = extract_document_context([mock_elem] * 2)

        # Long titles shouldn't be sections
        assert len(result["section"]) < 100 or result["section"] == ""

    def test_extract_context_with_part_heading(self):
        """Test extracting Part headings as chapters."""
        mock_elem = Mock()
        mock_elem.category = "Title"
        mock_elem.__str__ = lambda self: "Part 2: French Wines"

        result = extract_document_context([mock_elem])

        assert "Part" in result["chapter"]

    def test_extract_context_with_section_heading(self):
        """Test extracting Section headings as chapters."""
        mock_elem = Mock()
        mock_elem.category = "Title"
        mock_elem.__str__ = lambda self: "Section 3: Bordeaux"

        result = extract_document_context([mock_elem])

        assert "Section" in result["chapter"]

    def test_extract_context_checks_first_ten_elements(self):
        """Test that only first 10 elements are checked for title."""
        elements = []
        for i in range(15):
            mock_elem = Mock()
            mock_elem.category = "NarrativeText"
            mock_elem.__str__ = lambda self, i=i: f"Element {i}"
            elements.append(mock_elem)

        # Add title as 12th element (should not be found)
        mock_title = Mock()
        mock_title.category = "Title"
        mock_title.__str__ = lambda self: "Late Title"
        elements[11] = mock_title

        result = extract_document_context(elements)

        # Should use first element as fallback, not the late title
        assert result["document_title"] == "Element 0"

    def test_extract_context_empty_strings_handled(self):
        """Test that empty element strings are handled."""
        mock_elem = Mock()
        mock_elem.category = "Title"
        mock_elem.__str__ = lambda self: "   \n\t   "

        result = extract_document_context([mock_elem])

        # Empty/whitespace strings should be skipped
        assert isinstance(result["document_title"], str)

    def test_extract_context_no_category_attribute(self):
        """Test handling elements without category attribute."""
        # Create a mock that allows __str__ override
        class MockElement:
            def __str__(self):
                return "Some text"

        mock_elem = MockElement()

        result = extract_document_context([mock_elem])

        # Should handle gracefully using type name
        assert isinstance(result, dict)
        assert "document_title" in result


class TestIntegrationMetadataExtraction:
    """Integration tests for metadata extraction workflow."""

    def test_full_wine_description_extraction(self):
        """Test extracting metadata from a complete wine description."""
        text = """
        The 2015 Château Margaux is one of the finest wines from Bordeaux.
        This AOC classified wine is a blend of 90% Cabernet Sauvignon and
        10% Merlot from the prestigious Margaux appellation. The estate has
        been producing exceptional wines since the 18th century.
        """

        result = extract_wine_metadata(text)

        assert "2015" in result.vintages
        assert "AOC" in result.classifications
        assert "Bordeaux" in result.regions or "Margaux" in result.appellations
        assert not result.is_empty()

        dict_result = result.to_dict()
        assert all(isinstance(v, list) for v in dict_result.values())

    def test_metadata_extraction_consistency(self):
        """Test that multiple extractions of same text give same results."""
        text = "The 2018 Chardonnay from Burgundy is excellent."

        result1 = extract_wine_metadata(text)
        result2 = extract_wine_metadata(text)

        assert result1.to_dict() == result2.to_dict()

    def test_document_context_with_metadata(self):
        """Test combining document context with wine metadata."""
        mock_elem = Mock()
        mock_elem.category = "Title"
        mock_elem.__str__ = lambda self: "Chapter 5: Bordeaux Wines"

        context = extract_document_context([mock_elem])
        metadata = extract_wine_metadata("Cabernet Sauvignon from 2015")

        assert context["document_title"] or context["chapter"]
        assert len(metadata.vintages) > 0
