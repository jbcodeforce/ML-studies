# AI Solution Designs


**Summary**: An overview of various architectural patterns for deploying Gen AI, Hybrid AI, and Agentic AI solutions, ranging from basic private chatbots to complex RAG and stateful agentic systems.


**Sources**: /Users/jerome/Documents/Code/ML-studies/docs/architecture/sol-design.md


**Last updated**: 2026-04-18

---


This document presents different solution design patterns for deploying AI-driven applications, focusing on enterprise requirements such as security, scalability, and high availability.


## 1. Basic Private Chatbot (Cloud-Deployed)


Designed for B2B partnerships where end-users are not general consumers. Key requirements include Single Sign-On (SSO), high availability, and conversation tracing for quality assurance.


### Infrastructure & Routing


To achieve a highly available and scalable deployment, the architecture leverages the following:


- **Traffic Management**: [[amazon-route-53]] is used for global DNS routing, managing traffic across different regions and availability zones (AZs).


- **Load Balancing**: Within a region, [[aws-api-gateway]] and Elastic Load Balancers (ELB) manage traffic and routing.


- **Backend Services**: The architecture supports various compute backends, including [[aws-api-gateway]] routing to Lambda functions, SQS, or Fargate tasks in ECS.


- **Reliability**: The design emphasizes avoiding single points of failure, implementing backup/restore, disaster recovery (DR) failover, and using GitOps for deployment.


## 2. RAG (Retrieval-Augmented Generation) Architecture


A RAG solution extends the basic chatbot architecture by adding layers for document management and vector store management. This allows the model to retrieve context from private datasets to provide more accurate and relevant responses.


## 3. Stateful Agentic Applications


Advanced patterns for stateful agentic applications that manage complex, multi-step workflows and long-running processes.


## Related pages


- [[amazon-route-53]]
- [[aws-api-gateway]]
- [[rag-architecture]]
- [[po-processing]]
- [[agentic-ai-patterns]]