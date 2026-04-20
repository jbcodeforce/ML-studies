# Purchase Order Processing with AI and Hybrid Cloud


**Summary**: An architectural overview of an automated pipeline for extracting and interpreting data from unstructured purchase orders using Google Cloud services and agentic workflows.


**Sources**: /Users/jerome/Documents/Code/ML-studies/docs/architecture/po-processing.md


**Last updated**: 2026-04-18

---


This architecture implements a scope-reduced purchase order (PO) processing system for a manufacturing company. The goal is to automate the extraction and interpretation of information from unstructured submitted documents to improve upon existing rule-based software.


## The Processing Pipeline


The application follows a sequential, event-driven flow:


1.  **Ingestion**: POs are uploaded to [[google-cloud-storage]], organized by business dimensions like geography or customer name.
2.  **Event Notification**: An upload event triggers [[google-pub-sub]], which handles asynchronous processing and allows for scalable, decoupled event distribution.
3.  **Parsing & Extraction**: [[google-cloud-functions]] acts as the subscriber. It orchestrates the document parsing, splitting, and encoding. This step utilizes [[google-document-ai]] to transform unstructured PDFs or images into structured data.
4.  **Configuration Automation**: To handle complex product configurations, [[langgraph]] is used to implement a configuration tree. This manages conversation flow and interacts with expert systems via function calling.
5.  **Unstructured Request Handling**: [[google-gemini]] is integrated to support multi-modal entity extraction and agentic applications for handling unstructured user requests.


## Related pages


- [[google-cloud-storage]]
- [[google-pub-sub]]
- [[google-cloud-functions]]
- [[google-document-ai]]
- [[langgraph]]
- [[google-gemini]]
- [[product-configuration-automation]]