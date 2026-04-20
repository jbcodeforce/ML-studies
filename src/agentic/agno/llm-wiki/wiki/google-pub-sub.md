# Google Pub/Sub


**Summary**: A fully managed, asynchronous messaging service for real-time event ingestion and decoupled system architecture.


**Sources**: /Users/jerome/Documents/Code/ML-studies/docs/architecture/po-processing.md


**Last updated**: 2026-04-18

---


In the PO processing pipeline, [[google-pub-sub]] receives events from [[google-cloud-storage]] uploads. It acts as a buffer that authorizes asynchronous event processing and enables the system to scale by decoupling the ingestion of files from the processing logic (source: po-processing.md).


### Key Features


- **Scalability**: A distributed messaging system that can handle millions of messages per-second with low latency (source: po-processing.md).
- **Delivery Guarantees**: Provides at-most-once and at-least-once delivery, along with guaranteed message ordering (source: po-processing.md).
- **Topic-Based Mechanism**: Uses topics to categorize messages and route them to specific subscribers (source: po-processing.md).
- **Persistence**: Offers long-term persistence with replication to ensure no data loss (source: po-processing.md).
- **Cost Efficiency**: A pay-as-you-go model that scales automatically with message traffic spikes (source: po-processing.md).


## Related pages


- [[po-processing]]
- [[google-cloud-functions]]