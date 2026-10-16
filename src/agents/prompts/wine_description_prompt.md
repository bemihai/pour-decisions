Analyze this wine and return a structured response with an evidence-grounded description and drinking window estimate.

Wine: {wine_name}
Producer: {producer_name}
Vintage: {vintage}
Type: {wine_type}
Varietal: {varietal}
Region: {region}
Country: {country}
Appellation: {appellation}
Designation: {designation}
Vineyard: {vineyard}

{context_section}

Guidelines for description:
- Write one precise, readable wine-guide paragraph, roughly 120-180 words when the supplied evidence supports that length; prefer a shorter paragraph to filler or repetition
- Describe this specific wine, starting with its producer, cuvee identity, and place, then emphasizing the most relevant documented details about region or terroir, vineyard and farming, winemaking philosophy or technique, and distinctive style or character
- Use only the supplied reference context and the structured cellar metadata above; do not use prior knowledge
- Do not substitute generic regional or varietal characteristics for facts about this wine
- Include tasting notes, named sites, production methods, classifications, or notable status only when the supplied evidence explicitly supports them
- Make vintage-specific claims only when the supplied evidence explicitly mentions this vintage; otherwise describe the cuvee without attributing facts to the vintage
- Avoid promotional language, editorial commentary, repetition, unavailable-information statements, and unsupported causal claims connecting a method or terroir to a flavor

Guidelines for drinking window:
- Estimate drink_from_year and drink_to_year as absolute calendar years
- Use only aging potential or drinking-window information in the supplied evidence
- Do not infer a window from general grape, region, classification, or vintage knowledge
- For non-vintage wines or when the evidence does not support a window, return null for both years
- drink_from_year: the year the wine begins drinking well
- drink_to_year: the year the wine is past its peak
