---
title: "Private Chatbot Architecture"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/architecture/sol-design.md]
related: [ai-deployment-design, dns-routing-ai-services, rag-architecture]
tags: [chatbot, private-ai, websocket, authentication, high-availability, aws]
---

# Private Chatbot Architecture

A private chatbot deployed on a cloud provider must address enterprise-grade concerns including authentication, high availability, conversation tracing, and disaster recovery.

## Requirements
- **B2B-focused**: Not consumer-facing; users part of a B2B partnership
- **Authentication**: Login and Single Sign-On support
- **High availability**: Multi-region, multi-AZ deployment
- **Conversation tracing**: Store conversations in a lakehouse for quality assurance and AI accuracy enhancement

## Architecture Components

### Client Application
- Mobile app or reactive single-page web app
- Simple UI with login/authentication and chat interface
- **WebSocket connection** to conversation server for real-time communication

### Infrastructure
- **DNS routing**: Route 53 for region and AZ routing
- **Load balancing**: Elastic Load Balancers within regions and across AZs
- **Auto-scaling**: EC2 auto-scaling groups for the conversation server
- **API Gateway**: Optional layer for routing to Lambda, SQS, or ECS/Fargate
- **Static content**: S3 with CloudFront for CDN distribution
- **Disaster recovery**: Backup/restore, failover, and recovery planning
- **GitOps**: Infrastructure-as-code deployment management

### Key Design Decisions
- WebSocket-based communication for real-time chat (vs HTTP polling)
- Avoid single point of failure for all components
- Conversation tracing enables continuous quality improvement
- May use private VPC endpoints for security

## Sources
- [AI deployment solution designs](./sol-design.md)

## Related
- [AI Deployment Solution Designs](./ai-deployment-design.md)
- [DNS Routing and Load Balancing for AI Services](./dns-routing-ai-services.md)
- [RAG Architecture](./rag-architecture.md)