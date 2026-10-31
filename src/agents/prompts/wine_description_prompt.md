Analyze this wine and return a structured response with a description and drinking window estimate.

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

Guidelines when reference context is supplied:
- Write one precise, readable wine-guide paragraph, roughly 120-180 words when the supplied evidence supports that length; prefer a shorter paragraph to filler or repetition
- Describe this specific wine, starting with its producer, cuvee identity, and place, then emphasizing the most relevant documented details about region or terroir, vineyard and farming, winemaking philosophy or technique, and distinctive style or character
- Use only the supplied reference context and the structured cellar metadata above; do not use prior knowledge
- Do not substitute generic regional or varietal characteristics for facts about this wine
- Include tasting notes, named sites, production methods, classifications, or notable status only when the supplied evidence explicitly supports them
- Make vintage-specific claims only when the supplied evidence explicitly mentions this vintage; otherwise describe the cuvee without attributing facts to the vintage
- Avoid promotional language, editorial commentary, repetition, unavailable-information statements, and unsupported causal claims connecting a method or terroir to a flavor

Guidelines for general-knowledge fallback when no qualifying wine-specific evidence is available:
- Generate a useful, concise paragraph from the structured cellar metadata and general wine knowledge
- Explicitly state in the prose that the description is based on general knowledge of the grape, region, or style rather than evidence about this specific wine
- Describe typical grape, region, or style characteristics with qualified wording such as "typically", "often", or "may"; do not present them as observed tasting notes for this bottle
- Use only known cellar metadata to identify this wine; do not guess missing grape varieties or expand ambiguous cuvee abbreviations
- Do not invent wine-specific production methods, vineyard details, producer practices, awards, classifications, or vintage facts
- Prefer a shorter paragraph to filler; avoid promotional language and unsupported causal claims
- Return null for both drinking-window years

Guidelines for drinking window:
- Estimate drink_from_year and drink_to_year as absolute calendar years
- Use only aging potential or drinking-window information in the supplied evidence
- Do not infer a window from general grape, region, classification, or vintage knowledge
- For non-vintage wines or when the evidence does not support a window, return null for both years
- drink_from_year: the year the wine begins drinking well
- drink_to_year: the year the wine is past its peak
