# AI Deployment Solution Designs

This document presents examples of solution designs for deploying Gen AI, Hybrid AI, and Agentic AI systems in production environments.

## Main Thesis
The document outlines a progression of AI deployment patterns from basic private chatbots through RAG solutions, ML workflow orchestration, stateful agentic applications, and domain-specific solutions like purchase order processing. Each pattern adds architectural complexity and infrastructure requirements.

## Key Solutions

### 1. Basic Private Chatbot
A cloud-deployed chatbot for B2B use cases with:
- Single sign-on authentication
- High availability across regions and availability zones
- DNS routing via Route 53 with load balancing
- WebSocket-based real-time communication
- Conversation tracing in a lakehouse for quality assurance
- Auto-scaling, backup, disaster recovery, and GitOps

### 2. RAG Solution
Extends the chatbot with:
- Document management and ingestion
- Vector store for semantic retrieval
- Primarily targeted at internal enterprise use
- B2C exposure considered risky for regulated industries

### 3-4. ML Flow and Stateful Agentic Applications
Briefly mentioned without detailed architecture in this document.

### 5. Purchase Order Processing
A neuro-symbolic solution for partially automating purchase order processing and product configuration. References a dedicated architecture note.

## Infrastructure Concerns
- DNS routing and load balancing across regions/AZs
- API Gateway integration with Lambda, SQS, or ECS/Fargate backends
- Auto-scaling groups with Elastic Load Balancers
- Disaster recovery and failover planning
- GitOps for infrastructure management

## Connection to Wiki Concepts
This document informs several architectural concepts:
- [AI Deployment Solution Designs](../concepts/ai-deployment-design.md) — overall patterns and categories
- [Private Chatbot Architecture](../concepts/private-chatbot-architecture.md) — chatbot-specific architecture
- [DNS Routing and Load Balancing](../concepts/dns-routing-ai-services.md) — infrastructure concerns
- [RAG Architecture](../concepts/rag-architecture.md) — RAG-specific extension