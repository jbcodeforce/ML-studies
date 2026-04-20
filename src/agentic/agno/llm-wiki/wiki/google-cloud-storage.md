# Google Cloud Storage


**Summary**: A highly scalable and durable object storage service within the Google Cloud ecosystem.


**Sources**: /Users/jerome/Documents/Code/ML-studies/docs/architecture/po-processing.md


**Last updated**: 2026-04-18

---


[[google-cloud-storage]] is used in the PO processing pipeline to host uploaded purchase orders, organized by business dimensions such as geography or customer name (source: po-processing.md).


### Key Features


- **Scalability & Durability**: Provides 11 9s of availability (source: po-processing.md).
- **Storage Classes**: Offers various classes to optimize for performance and cost requirements (source: po-processing.md).
- **Global Availability**: Replicated across locations for high availability and low-latency access worldwide (source: po-processing.md).
- **Security**: Implements server-side encryption and IAM-based access control (source: po-processing.md).
- **Data Integrity**: Supports object versioning and the ability to restore to previous versions (source: po-processing.md).


## Related pages


- [[po-processing]]