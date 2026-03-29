Analyze this wine and return a structured response with a brief description and drinking window estimate.

Wine: {wine_name}
Producer: {producer_name}
Vintage: {vintage}
Type: {wine_type}
Varietal: {varietal}
Region: {region}
Country: {country}
Appellation: {appellation}

{context_section}

Guidelines for description:
- Write 2-3 sentences focusing on expected flavor profile, style, and character
- If reference context is provided, prioritize information from it
- Keep it factual based on typical regional and varietal characteristics
- Do not invent specific tasting notes

Guidelines for drinking window:
- Estimate drink_from_year and drink_to_year as absolute calendar years
- Base the estimate on grape variety aging potential, region, classification, and vintage year
- If reference context mentions aging potential or drinking windows, use that information
- For non-vintage wines or when genuinely uncertain, return null for both years
- drink_from_year: the year the wine begins drinking well
- drink_to_year: the year the wine is past its peak
