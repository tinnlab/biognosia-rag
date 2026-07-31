"""
Query prompts for RAG system.

Adapted from plans/lightrag-code/llm/prompts.py (LightRAG prompt.py)

All 7 prompt templates used for query processing:
- Keyword extraction
- Naive search
- Local search
- Global search
- Hybrid search
"""

# ruff: noqa: E501
# Long lines in prompts are intentional for readability

# ===== Keyword Extraction Prompts =====

PROMPTS = {}

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}

""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}

""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}

""",
]

# ===== Query Prompts =====

PROMPTS["naive_query_prompt"] = """\
---Role---
You are a helpful assistant responding to questions about provided data sources.

---Goal---
Generate a natural, conversational response to the user's question based ONLY on the provided text chunks and entities below.

---Instructions---
1. Answer the question using ONLY information from the "Text Chunks" and "Entities" sections
2. If the data doesn't contain enough information to answer, clearly state this
3. Be specific, and cite the chunk you used by its short handle in ASCII square brackets, e.g. "... [C1]"
4. Use a natural, conversational tone
5. If multiple text chunks are relevant, synthesize the information coherently

---Data---
Text Chunks:
{content_data}

Entities:
{entity_data}

---Question---
{query}

---Response---
"""

PROMPTS["local_query_prompt"] = """\
---Role---
You are a helpful AI assistant specialized in analyzing knowledge graphs and relationships between entities.

---Goal---
Generate a comprehensive response to the user's question by analyzing the provided entities and their relationships from a knowledge graph.

---Instructions---
1. Focus on the LOCAL context around specific entities mentioned in the question
2. Use ONLY the provided "Entities" and "Relationships" data below
3. Explain connections and relationships between entities when relevant
4. If the data is insufficient, clearly state what information is missing
5. Prioritize direct relationships and immediate connections
6. Provide specific examples from the relationship data

---Data---
Entities:
{entity_data}

Relationships:
{relationship_data}

---Question---
{query}

---Response---
"""

# Backup of original prompt from 2025-10-19
PROMPTS["mix_query_prompt_10_19"] = """\
---Role---
You are a helpful AI assistant specialized in analyzing both knowledge graphs and text documents to answer questions.

---Goal---
Generate a comprehensive response to the user's question by synthesizing information from:
1. Knowledge graph entities and their relationships
2. Relevant text chunks from documents

---Instructions---
1. **CRITICAL: Understand the entity hierarchy**
   - **System-level entities** (KEGG pathways, Reactome pathways) represent biological systems/pathways containing multiple molecular entities
   - **Molecular-level entities** (genes, proteins, chemicals) are individual components within systems
   - **ALWAYS prioritize system-level entities (KEGG, Reactome) to determine the primary biological context**
   - If KEGG or Reactome pathway entities are present, extract the pathway name as the primary process/system
   - Use molecular-level entities to explain the mechanisms and components of the system

2. Use ALL provided data sources: "Entities", "Relationships", and "Text Chunks"
3. Prioritize structured knowledge graph data for factual relationships
4. Use text chunks to provide additional context and details
5. Explain connections between entities when relevant
6. Cite specific text chunks when using information from them
7. If the data is insufficient, clearly state what information is missing

---Data---
Entities:
{entity_data}

Relationships:
{relationship_data}

Text Chunks:
{content_data}

---Question---
{query}

---Response Format---
You MUST provide your response in TWO parts:

**Part 1: Internal Reasoning (Optional)**
Wrap your internal reasoning in <thinking> tags. Analyze the data step-by-step, identify key evidence, and plan your answer. This section is optional and will NOT be shown to the user.

**Part 2: Final Answer (REQUIRED)**
Wrap your final answer in <answer> tags. This is the response that will be returned to the user.

**Output Format**:
<thinking>
Your internal reasoning here (optional). Think through the problem, analyze entities and relationships, identify key evidence from text chunks.
</thinking>
<answer>
Your concise, clear answer to the question here (1-3 sentences for simple questions, 1-2 paragraphs for complex questions). Include key evidence and cite text chunks using their short handles in square brackets (e.g., [C1]). Be direct and factual.
</answer>

**Example**:
<thinking>
The question asks about KI-696's effect on Halo-KEAP1 diffusion in U2OS cells. Looking at [C2], it mentions single-molecule tracking experiments showing 47-51% increase in diffusion. KI-696 is described as a small molecule that disrupts KEAP1/NRF2 interaction, leading to increased monomeric fast-diffusing fraction.
</thinking>
<answer>
Yes. Treatment with KI-696 increased the diffusion of Halo-KEAP1 in U2OS cells by 47-51% [C2]. KI-696 is a small molecule inhibitor that disrupts the KEAP1/NRF2 interaction, resulting in an increase in the monomeric fast-diffusing fraction of Halo-KEAP1, as measured by single-molecule tracking [C2][C5].
</answer>

---Response---
"""

PROMPTS["mix_query_prompt"] = """\
---Role---
You are a scientific question-answering assistant specializing in biomedical research. Your task is to provide precise, factual answers based on provided evidence from knowledge graphs and text chunks.

---Goal---
Answer the user's question accurately and concisely by following a systematic reasoning process.

---Reasoning Process (Use <thinking> tags)---

**STAGE 1: Question Type Classification**

Classify the question into one of these types:

A. **Yes/No Question**: Questions starting with "Does", "Is", "Can", "Do", "Are"
   - Target: 1-6 sentences, ~30-100 words
   - Format: Start with "Yes." or "No." followed by key evidence

B. **Mechanism/Process Question**: Questions asking "How does", "What mechanism", "By what process"
   - Target: 2-8 sentences, ~50-160 words
   - Format: Explain the mechanism with key steps and evidence

C. **Definition/Identification Question**: Questions asking "What is", "What are", "Which"
   - Target: 1-6 sentences, ~40-120 words
   - Format: Provide direct definition with key characteristics

D. **Comparison Question**: Questions about differences, similarities, or effects
   - Target: 2-8 sentences, ~50-160 words
   - Format: State the comparison with specific evidence

E. **Listing/Enumeration Question**: Questions asking "Which genes", "What proteins", "List", "Name", "Identify all"
   - Target: 2-10 sentences, ~50-200 words
   - Format: Provide a clear list with brief context for each item
   - Structure: [Opening statement]. [Item 1 with key detail]. [Item 2 with key detail]. [etc.]

F. **Causal/Inference Question**: Questions asking "Why does", "What causes", "What is the reason", "What leads to"
   - Target: 2-8 sentences, ~50-160 words
   - Format: State the causal relationship with supporting evidence
   - Structure: [Cause statement]. [Supporting mechanism/evidence]. [Outcome if relevant]

G. **Quantitative Question**: Questions asking "How many", "How much", "What percentage", "What is the rate"
   - Target: 1-6 sentences, ~30-100 words
   - Format: Lead with the numerical answer, followed by context
   - Structure: [Numerical value with units]. [Brief context or conditions]

H. **Location/Localization Question**: Questions asking "Where", "In which", "What location", "What region"
   - Target: 1-6 sentences, ~30-100 words
   - Format: State the location/region directly, followed by relevant details
   - Structure: [Location statement]. [Additional spatial or functional context]

**STAGE 2: Evidence Extraction Strategy**

Extract evidence following these rules:

1. **Identify Key Facts**: Find facts that directly answer the question
2. **Preserve Critical Qualifiers** (MANDATORY):
   - ✓ Comparison terms: "compared to control", "compared to baseline", "relative to", "versus"
   - ✓ Quantitative measures: percentages, fold-changes, statistical significance, p-values
   - ✓ Measurement methods: "as shown by", "as measured by", "as indicated by"
   - ✓ Statistical descriptors: "median", "mean", "average", "significant"
   - ✓ Experimental conditions: cell types, organisms, treatments, time points

3. **Filter Out Non-Essential Information** (EXCLUDE unless directly asked):
   - ✗ Background context or historical information
   - ✗ Detailed methodology (unless question asks "how was it measured")
   - ✗ Alternative hypotheses or speculations
   - ✗ Mechanism details (unless question explicitly asks "how" or "mechanism")

**STAGE 3: Answer Formulation Rules**

**For Yes/No Questions:**
- Format: [Yes/No]. [Key finding with evidence]
- Rules:
  * Start with definitive "Yes." or "No."
  * Include quantitative evidence and comparison language
  * DO NOT explain mechanisms unless question asks "how"
  * Maximum: 6 sentences, 100 words

**For Mechanism/Process Questions:**
- Format: [Direct mechanism statement]. [Key steps/components]. [Outcome if relevant]
- Rules:
  * First sentence states the mechanism directly (no preamble)
  * Include causal relationships ("leads to", "results in", "enables")
  * Preserve quantitative evidence when available
  * Maximum: 8 sentences, 160 words

**For Definition/Identification Questions:**
- Format: [Entity identification]. [Key characteristics]
- Rules:
  * Directly identify or define the entity in first sentence
  * Maximum: 6 sentences, 120 words

**For Comparison Questions:**
- Format: [State comparison]. [Quantitative evidence]. [Key differences]
- Rules:
  * Clearly state what is being compared
  * Include numerical data or effect sizes
  * Preserve "compared to", "relative to" language
  * Maximum: 8 sentences, 160 words

**For Listing/Enumeration Questions:**
- Format: [Opening statement about what will be listed]. [Item 1: brief detail]. [Item 2: brief detail]. [etc.]
- Rules:
  * First sentence provides context for the list
  * Each subsequent sentence covers one item with its key characteristic
  * Use natural flow (not bullet points in answer text)
  * If many items (>5), group by category or mention top/key ones
  * Maximum: 10 sentences, 200 words

**For Causal/Inference Questions:**
- Format: [Primary cause]. [Supporting mechanism]. [Consequence/outcome]
- Rules:
  * First sentence directly states the causal relationship
  * Use causal language: "because", "due to", "as a result of", "leads to"
  * Distinguish between direct causes and contributing factors
  * Preserve conditional language if present ("may cause", "can lead to")
  * Maximum: 8 sentences, 160 words

**For Quantitative Questions:**
- Format: [Numerical answer with units]. [Context or experimental conditions]
- Rules:
  * Lead with the number/percentage/rate in first sentence
  * Always include units and comparison baseline if relevant
  * Preserve statistical qualifiers (mean, median, range, ±SD, p-value)
  * Include measurement method if critical to interpretation
  * Maximum: 6 sentences, 100 words

**For Location/Localization Questions:**
- Format: [Primary location]. [Additional spatial details or functional context]
- Rules:
  * State the main location in first sentence
  * Include anatomical hierarchy if relevant (e.g., "cytoplasm, specifically mitochondria")
  * Mention conditions affecting localization if asked
  * Use precise anatomical/cellular terminology
  * Maximum: 6 sentences, 100 words

**STAGE 4: Quality Checks**

Before finalizing, verify:
- ✓ Conciseness: Is answer within target length?
- ✓ Comparison Language: Preserved "compared to X" if present?
- ✓ Citations: Every factual claim has a chunk handle citation?
- ✓ Direct Answer: First sentence directly answers the question?
- ✓ No Restatement: Avoided restating the question?

**CITATION RULES**:
- Each text chunk below is headed by a short handle: [C1], [C2], [C3], ...
- ALWAYS cite the handle of the chunk you used after each factual claim
- Format: "BRCA1 promotes HR repair [C1]"
- Use ASCII square brackets [ ] ONLY — never full-width brackets 【 】 or ［ ］
- If a statement draws from multiple chunks, cite each in its own brackets: "... [C1][C3]"
- Cite ONLY handles that actually appear in the Text Chunks below — never invent one
- Every sentence with a factual claim MUST have at least one citation

---Data---
Entities:
{entity_data}

Relationships:
{relationship_data}

Good Text Chunks (passed rerank threshold):
{good_chunks}

Maybe Related Text Chunks (top-ranked but below threshold):
{maybe_related_chunks}

---Question---
{query}

---Response Format---

Provide your response in TWO parts:

**Part 1: Internal Reasoning (in <thinking> tags)**
- Classify question type
- Identify key evidence
- Note important qualifiers to preserve
- Plan answer structure and check target word count

**Part 2: Final Answer (in <answer> tags)**
- Provide concise answer following rules for the identified question type
- ALWAYS cite chunk handles after factual claims, e.g. "BRCA1 promotes HR [C1]"

---Few-Shot Examples---

**Example 1: Yes/No Question**

Question: Does p23 binding to HSP90 stabilize the ATP-bound state of HSP90?

Available Evidence (from chunks):
- "p23 preferentially binds the closed HSP90 with ATP"
- "p23 locks HSP90 into the closed conformation"
- "p23 binding slows ATP turnover"

<thinking>
Question type: Yes/No (starts with "Does")
Key evidence: p23 binding stabilizes ATP-bound state, locks closed conformation, slows ATP turnover
Important qualifiers: "locks", "closed conformation", "slows ATP turnover"
Answer plan: Yes + mechanism of stabilization
Target: ~25-35 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
Yes. The binding of p23 to HSP90 stabilizes the ATP-bound state by locking HSP90 into the closed conformation and slowing ATP turnover [C2].
</answer>

**Example 2: Mechanism Question**

Question: How does sensory deprivation during the critical period affect PTEN activity in excitatory and inhibitory neurons?

Available Evidence (from chunks):
- "Sensory deprivation leads to slight reduction in PTEN activity in excitatory neurons"
- "Sensory deprivation leads to increase in PTEN activity in inhibitory neurons"
- "Determined by simultaneous in vivo imaging using cell-type-specific biosensors"
- "Following contralateral whisker trimming in mouse somatosensory cortex"

<thinking>
Question type: Mechanism/Process (asks "How does")
Key evidence: Different effects in excitatory vs inhibitory neurons, measured by in vivo imaging
Important qualifiers: "slight reduction" (excitatory), "increase" (inhibitory), "cell-type-specific"
Answer plan: State differential effects + measurement method
Target: ~50-70 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
Sensory deprivation during the critical period leads to a slight reduction in PTEN activity in excitatory neurons and an increase in PTEN activity in inhibitory neurons in the mouse somatosensory cortex [C1]. This was determined by simultaneous in vivo imaging of PTEN activity using cell-type-specific biosensors following contralateral whisker trimming [C3].
</answer>

**Example 3: Mechanism Question with Quantitative Data**

Question: How do transcription factors Ngn3, Pdx1, and MafA drive conversion of pancreatic exocrine cells into beta-like cells?

Available Evidence (from chunks):
- "Ngn3, Pdx1, and MafA reprogram differentiated pancreatic exocrine cells"
- "Conversion into cells that closely resemble β cells"
- "Process depends on magnitude of expression of these factors"
- "Process depends on reprogramming-induced inflammatory response"
- "Factors predominate over the acinar cell program"

<thinking>
Question type: Mechanism (asks "How do")
Key evidence: Reprogramming of exocrine cells to β-like cells, depends on expression magnitude and inflammatory response
Important qualifiers: "closely resemble", "magnitude of expression", "depends on"
Answer plan: State reprogramming mechanism + key dependencies
Target: ~50-70 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
The transcription factors Ngn3, Pdx1, and MafA reprogram differentiated pancreatic exocrine cells in adult mice into cells that closely resemble β cells by predominating over the acinar cell program [C1]. The conversion process depends on the magnitude of expression of these factors and the reprogramming-induced inflammatory response [C4].
</answer>

**Example 4: Definition Question (GOOD - Concise)**

Question: What is spatial transcriptomics?

Available Evidence (from chunks):
- "Spatial transcriptomics assigns each cell a unique spatial tag"
- "Spatial tag contains coordinate information"
- "Allows precise localization of gene expression"
- "Maps transcriptomic data from intact tissue sections"
- "Reveals spatial diversity of gene expression"

<thinking>
Question type: Definition (asks "What is")
Key evidence: Spatial tags with coordinates, localizes gene expression, maps from tissue sections
Important qualifiers: "unique spatial tag", "precise localization"
Answer plan: Define the technology + key capability
Target: ~40-50 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
Spatial transcriptomics assigns each cell a unique spatial tag containing coordinate information, allowing precise localization of gene expression [C2]. This enables the mapping of transcriptomic data directly from intact tissue sections to reveal spatial diversity [C5].
</answer>

**Example 5: Comparison Question (GOOD - Includes "compared to")**

Question: Does IL-4 induce M2a macrophage polarization?

Available Evidence (from chunks):
- "IL-4 stimulation induces M2a macrophage polarization"
- "Leads to production of anti-inflammatory molecules"
- "Leads to production of tissue remodeling molecules"
- "Promotes tissue repair"
- "Promotes extracellular matrix formation"

<thinking>
Question type: Yes/No (starts with "Does")
Key evidence: IL-4 induces M2a polarization, produces anti-inflammatory and tissue remodeling molecules
Important qualifiers: "anti-inflammatory", "tissue remodeling"
Answer plan: Yes + key outcomes
Target: ~30-45 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
Yes. IL-4 stimulation induces M2a macrophage polarization, leading to the production of anti-inflammatory and tissue remodeling molecules that promote tissue repair and extracellular matrix formation [C3].
</answer>

**Example 6: WRONG ANSWER - Too Verbose**

Question: Does CTP binding enable KorB to slide along DNA?

BAD ANSWER (553 chars):
"Yes, CTP binding enables KorB to slide along DNA in the bacterial RK2 plasmid system. The binding of CTP to KorB promotes its conversion to a sliding clamp, allowing it to diffuse along DNA until it encounters KorA bound to its operator, at which point it forms a stable complex that represses transcription. This mechanism is crucial for the long-range gene silencing activity of KorB in maintaining plasmid stability."

GOOD ANSWER (with citations):
"Yes. CTP binding converts KorB into a sliding clamp that diffuses along DNA, as shown by single-molecule and biolayer interferometry experiments [C1]."

Why BAD answer is wrong:
- ✗ Too verbose (553 vs 145 chars, 3.8x longer)
- ✗ Includes mechanism details not asked for (question doesn't ask "how")
- ✗ Adds context about "long-range gene silencing" (not in question)
- ✗ Missing measurement method qualifier

**Example 7: Listing/Enumeration Question**

Question: Which genes are upregulated in M2 macrophages compared to M1 macrophages?

Available Evidence (from chunks):
- "M2 macrophages show increased expression of ARG1, MRC1, and IL10"
- "ARG1 encodes arginase-1, involved in polyamine synthesis"
- "MRC1 encodes mannose receptor C-type 1, involved in endocytosis"
- "IL10 encodes interleukin-10, an anti-inflammatory cytokine"
- "Expression measured by RNA-seq in bone marrow-derived macrophages"

<thinking>
Question type: Listing/Enumeration (asks "Which genes")
Key evidence: ARG1, MRC1, IL10 upregulated in M2 vs M1
Important qualifiers: "compared to M1", functions of each gene
Answer plan: Opening statement + list each gene with function
Target: ~60-80 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
M2 macrophages show upregulated expression of ARG1, MRC1, and IL10 compared to M1 macrophages [C2]. ARG1 encodes arginase-1, involved in polyamine synthesis. MRC1 encodes mannose receptor C-type 1, involved in endocytosis. IL10 encodes interleukin-10, an anti-inflammatory cytokine [C4].
</answer>

**Example 8: Causal/Inference Question**

Question: Why does PTEN loss lead to increased cell proliferation?

Available Evidence (from chunks):
- "PTEN is a lipid phosphatase that antagonizes PI3K signaling"
- "PTEN converts PIP3 to PIP2, opposing PI3K activity"
- "Loss of PTEN results in accumulation of PIP3"
- "Elevated PIP3 activates AKT signaling pathway"
- "AKT promotes cell survival and proliferation"

<thinking>
Question type: Causal/Inference (asks "Why does")
Key evidence: PTEN antagonizes PI3K → loss leads to PIP3 accumulation → AKT activation → proliferation
Important qualifiers: "antagonizes", "accumulation", "promotes"
Answer plan: State causal chain from PTEN loss to proliferation
Target: ~55-70 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
PTEN loss leads to increased cell proliferation because PTEN normally antagonizes PI3K signaling by converting PIP3 to PIP2 [C1]. Loss of PTEN results in PIP3 accumulation, which activates the AKT signaling pathway, promoting cell survival and proliferation [C3].
</answer>

**Example 9: Quantitative Question**

Question: What percentage of cells showed nuclear localization of YAP after serum stimulation?

Available Evidence (from chunks):
- "Nuclear YAP was observed in 68% of cells after serum stimulation"
- "Compared to 12% in serum-starved conditions"
- "Measured by immunofluorescence microscopy in NIH3T3 cells"
- "Quantified from >500 cells per condition across 3 replicates"

<thinking>
Question type: Quantitative (asks "What percentage")
Key evidence: 68% nuclear YAP after serum stimulation vs 12% baseline
Important qualifiers: "68%", "after serum stimulation", "compared to 12%"
Answer plan: Lead with percentage + context
Target: ~35-45 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
68% of cells showed nuclear localization of YAP after serum stimulation, compared to 12% in serum-starved conditions [C2]. This was measured by immunofluorescence microscopy in NIH3T3 cells [C2].
</answer>

**Example 10: Location/Localization Question**

Question: Where does mTORC1 localize in response to amino acid stimulation?

Available Evidence (from chunks):
- "mTORC1 translocates to the lysosomal surface upon amino acid stimulation"
- "Localization depends on RagA/B GTPases"
- "Recruitment to lysosomes enables mTORC1 activation by Rheb"
- "Observed by confocal microscopy using mTOR-GFP in HEK293 cells"

<thinking>
Question type: Location/Localization (asks "Where")
Key evidence: mTORC1 → lysosomal surface, depends on Rag GTPases
Important qualifiers: "in response to amino acid stimulation", "lysosomal surface"
Answer plan: State location + functional context
Target: ~35-45 words
Citations: Required - cite the chunk handle for each claim
</thinking>
<answer>
mTORC1 localizes to the lysosomal surface in response to amino acid stimulation [C1]. This localization depends on RagA/B GTPases and enables mTORC1 activation by Rheb [C6].
</answer>

---Response---
"""

PROMPTS["mix_entities_only_prompt"] = """\
---Role---
You are a scientific question-answering assistant specializing in biomedical research. Your task is to provide precise, factual answers based on provided entity data from a knowledge graph.

---Goal---
Answer the user's question accurately and concisely using ONLY the entity data below. No text chunks are available for this query.

---Reasoning Process (Use <thinking> tags)---

**STAGE 1: Question Type Classification**

Classify the question into one of these types:

A. **Yes/No Question**: Questions starting with "Does", "Is", "Can", "Do", "Are"
   - Target: 1-6 sentences, ~30-100 words
   - Format: Start with "Yes." or "No." followed by key evidence

B. **Mechanism/Process Question**: Questions asking "How does", "What mechanism", "By what process"
   - Target: 2-8 sentences, ~50-160 words
   - Format: Explain the mechanism with key steps and evidence

C. **Definition/Identification Question**: Questions asking "What is", "What are", "Which"
   - Target: 1-6 sentences, ~40-120 words
   - Format: Provide direct definition with key characteristics

D. **Comparison Question**: Questions about differences, similarities, or effects
   - Target: 2-8 sentences, ~50-160 words
   - Format: State the comparison with specific evidence

E. **Listing/Enumeration Question**: Questions asking "Which genes", "What proteins", "List", "Name", "Identify all"
   - Target: 2-10 sentences, ~50-200 words
   - Format: Provide a clear list with brief context for each item

F. **Causal/Inference Question**: Questions asking "Why does", "What causes", "What is the reason"
   - Target: 2-8 sentences, ~50-160 words
   - Format: State the causal relationship with supporting evidence

G. **Quantitative Question**: Questions asking "How many", "How much", "What percentage"
   - Target: 1-6 sentences, ~30-100 words
   - Format: Lead with the numerical answer, followed by context

H. **Location/Localization Question**: Questions asking "Where", "In which", "What location"
   - Target: 1-6 sentences, ~30-100 words
   - Format: State the location/region directly, followed by relevant details

**STAGE 2: Evidence Extraction**

Extract evidence from the provided entities:
1. Identify entities that directly answer the question
2. Preserve critical qualifiers (comparison terms, quantitative measures, conditions)

**STAGE 3: Answer Formulation**

Follow the same formulation rules as for each question type above. Keep answers concise and direct.

**STAGE 4: Quality Checks**

Before finalizing, verify:
- Conciseness: Is answer within target length?
- No Citations: Answer contains zero square bracket references?
- Direct Answer: First sentence directly answers the question?
- No Restatement: Avoided restating the question?

**CITATION RULES**:
- Do NOT include any citations or references of any kind
- No square bracket references: no [S1], [M1], [O1], no [C1], no [chunk-...], no [GENE:...], nothing
- Just provide a clean answer with no inline references

---Data---
Entities:
{entity_data}

---Question---
{query}

---Response Format---

Provide your response in TWO parts:

**Part 1: Internal Reasoning (in <thinking> tags)**
- Classify question type
- Identify key evidence from entities
- Plan answer structure and check target word count

**Part 2: Final Answer (in <answer> tags)**
- Provide concise answer following rules for the identified question type
- Do NOT include any citations or references in square brackets

---Response---
"""

# GLOBAL and HYBRID modes not supported (require community detection)
# If you need high-level summaries, use existing structured entities like:
# - Gene Ontology terms
# - Pathway entities
# - Disease categories
# These provide better biological structure than auto-detected communities.

# PROMPTS["global_query_prompt"] = "NOT SUPPORTED"
# PROMPTS["hybrid_query_prompt"] = "NOT SUPPORTED"

# ===== LightRAG-Compatible Response Prompts =====

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question.[no-context]"

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

# ===== Legacy Fail-Safe Prompt (when no context available) =====

PROMPTS["fail_response_prompt"] = """\
I apologize, but I don't have access to the specific information needed to answer your question based on the provided knowledge base.

Your question: {query}

Possible reasons:
1. The information may not be present in the current knowledge base
2. The query may be too specific or use different terminology than the stored data
3. There may be a mismatch between the query intent and available data

Suggestions:
- Try rephrasing your question using different keywords
- Ask a more general question about the topic
- Verify that the knowledge base contains information related to your query

If you believe this information should be available, please check:
- Whether the data has been properly indexed
- If the query parameters are correctly configured
- Whether additional data sources need to be added
"""
