<div align="center">

# 🛡️ MedBuddy Clinical RAG
### Grounded Medical Decision Support & Citation Synthesis Engine

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1200&color=38BDF8&center=true&vCenter=true&width=750&lines=MedBuddy+Clinical+RAG;Evidence-Grounded+Medical+Reasoning;Amazon+Bedrock+Claude+3.5+%2B+OpenSearch;Verifiable+Clinical+Citations+with+Zero+Hallucination" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AWS-Bedrock_Claude_3.5-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white" />
  <img src="https://img.shields.io/badge/Vector_DB-OpenSearch_Serverless-0052CC?style=for-the-badge&logo=opensearch&logoColor=white" />
  <img src="https://img.shields.io/badge/Retrieval-Dense_%2B_BM25_Hybrid-22C55E?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Guardrail-TruLens_Factuality-E11D48?style=for-the-badge" />
</p>

<img src="https://skillicons.dev/icons?i=python,aws,docker,fastapi,postgres&theme=dark" alt="Stack" />

</div>

---

### 🔬 Grounded Query Execution

```bash
┌──(prak05@medbuddy-rag)-[~/query]
└─$ curl -X POST https://api.medbuddy.aws.internal/v1/clinical/ask \
     -d '{"question": "What is first-line pharmacological treatment for uncomplicated Stage 1 hypertension with microalbuminuria?"}'
```

```json
{
  "answer": "First-line therapy consists of an ACE inhibitor (such as Lisinopril 10-20mg once daily) or an ARB (such as Losartan 50mg once daily) due to demonstrated renal-protective benefits in microalbuminuria.",
  "citations": [
    { "guideline": "ACC/AHA 2024 Hypertension Guidelines", "section": "Table 8.4", "evidence_quote": "ACE inhibitors or ARBs are recommended as first-line in patients with CKD or albuminuria." }
  ],
  "hallucination_score": 0.02,
  "confidence": 0.98,
  "retrieval_method": "Hybrid Dense-Cosine (0.6) + Lexical BM25 (0.4)"
}
```

---

## 🏛️ RAG Engine Bento Grid

<table width="100%">
  <tr>
    <td width="50%" valign="top">
      <h3>🔍 Hybrid Dual-Pass Retrieval</h3>
      <p>Fuses <b>dense semantic vector embeddings</b> (Titan Text Embeddings V2) with exact <b>BM25 lexical scoring</b> to preserve critical pharmacological dosages and drug trade names.</p>
      <b>Features:</b> Reciprocal Rank Fusion (RRF) • Medical acronym resolution
    </td>
    <td width="50%" valign="top">
      <h3>📚 Verified Evidence Citations</h3>
      <p>Every clinical assertion is grounded by verifiable paragraph and table offsets. Responses failing source verification are automatically rejected before delivery.</p>
      <b>Features:</b> Zero hallucination safeguard • Source offset pointers
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>⚡ Amazon OpenSearch Serverless</h3>
      <p>Serverless vector search graph handling indexing and low-latency KNN queries across thousands of peer-reviewed clinical guidelines and trial documents.</p>
      <b>Features:</b> Automatic scaling • Zero cluster management
    </td>
    <td width="50%" valign="top">
      <h3>🩺 Multi-Turn Context Windows</h3>
      <p>Stateful patient case analysis allowing clinicians to drill down into comorbid conditions, lab interactions, and contraindication profiles dynamically.</p>
      <b>Features:</b> DynamoDB session state • Case timeline tracking
    </td>
  </tr>
</table>

---

## ⚡ Run Locally

```bash
git clone https://github.com/prak05/MedBuddy-Clinical-RAG.git
cd MedBuddy-Clinical-RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000
```

---

<div align="center">
<sub>Preserved Baseline: <a href="https://github.com/prak05/MedBuddy-Clinical-RAG/tree/Ex">branch Ex</a> • Engineered for Scalable Production Clouds</sub>
</div>
