# MedBuddy — Clinical Decision Support & Medical Education RAG

[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock%20Claude%203.5-FF9900?logo=amazon-aws&logoColor=white)](#aws-cloud-architecture)
[![RAG System](https://img.shields.io/badge/Architecture-Hybrid%20Dense%2FBM25%20RAG-0052CC)](#rag-architecture)
[![Branch Status](https://img.shields.io/badge/Preservation-Ex%20Branch-22C55E)](#branch-architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-gray.svg)](LICENSE)

> Grounded medical reasoning and clinical guideline synthesis platform powered by Amazon Bedrock, OpenSearch Serverless, and verifiable evidence citations.

---

## 1. Executive Brief
Clinical inquiry requires deterministic accuracy and verifiable citations to peer-reviewed literature. **MedBuddy** implements a hybrid Retrieval-Augmented Generation (RAG) system combining dense vector embeddings with lexical BM25 retrieval to synthesize clinical answers without factual hallucinations.

* **Consolidation**: Merged IBM Watsonx prototype into the canonical enterprise Bedrock stack.
* **Preservation Snapshot**: Original IBM Watson & LangChain prototypes preserved on [`Ex`](https://github.com/prak05/MedBuddy-Clinical-RAG/tree/Ex).
* **Target Architecture**: AWS Bedrock, OpenSearch Serverless, and ECS Fargate.

---

## 2. AWS Cloud Architecture

```mermaid
flowchart TD
    Clinician["Clinician / Student"] --> CloudFront["Amazon CloudFront (CDN)"]
    CloudFront --> S3Web["S3 Static Portal"]
    Clinician --> APIGW["Amazon API Gateway (REST / WebSocket)"]
    APIGW --> Cognito["Amazon Cognito (HIPAA / RBAC Auth)"]
    Cognito --> ECS["AWS ECS Fargate (FastAPI RAG Engine)"]
    
    subgraph Knowledge Base
        S3Docs["Amazon S3 (Clinical Guidelines Vault)"]
        OpenSearch["Amazon OpenSearch Serverless (Hybrid Vector + BM25)"]
    end
    
    subgraph AI Inference
        BedrockLLM["Amazon Bedrock (Anthropic Claude 3.5 Sonnet)"]
        BedrockEmbed["Amazon Titan Text Embeddings V2"]
    end
    
    S3Docs --> BedrockEmbed
    BedrockEmbed --> OpenSearch
    ECS --> OpenSearch
    ECS --> BedrockLLM
    ECS --> Dynamo["Amazon DynamoDB (Session History & Audit Trail)"]
```

---

## 3. Engineering Improvements
1. **Hybrid Retrieval**: Combines semantic cosine similarity with BM25 keyword matching for medical acronyms and pharmacological dosages.
2. **Hallucination Guardrails**: Implements TruLens-style factuality validation scoring where citations are cross-referenced with source text offsets before streaming responses.
3. **Data Security**: AWS KMS envelope encryption for all in-transit and at-rest clinical queries.

---

## 4. Run & Deployment Runbook

### Local Quickstart
```bash
git clone https://github.com/prak05/MedBuddy-Clinical-RAG.git
cd MedBuddy-Clinical-RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

---

## 5. Branch Architecture
* [`main`](https://github.com/prak05/MedBuddy-Clinical-RAG/tree/main): Enterprise Bedrock RAG platform with OpenSearch and FastAPI.
* [`Ex`](https://github.com/prak05/MedBuddy-Clinical-RAG/tree/Ex): Preserved historical baseline.

<!-- antigravity-certified -->
