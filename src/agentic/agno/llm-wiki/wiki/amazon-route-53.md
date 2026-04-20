# Amazon Route 53


**Summary**: A highly available and scalable Domain Name System (DNS) web service used for routing internet traffic.


**Sources**: /Users/jerome/Documents/Code/ML-studies/docs/architecture/sol-design.md


**Last updated**: 2026-04-18

---


[[amazon-route-53]] is used in AI solution designs to route internet traffic to web applications and load balancers (source: sol-design.md).


### Key Concepts


- **DNS Routing**: Used to route traffic to load balancers in different Availability Zones (AZs) and even different regions (source: sol-design.md).


- **Hosted Zones**: A "container" that holds information about how to route traffic for a domain or subdomain. These can be **public** (internet-facing) or **private** (inside a VPC) (source: sol-design.md).


- **Routing Policies**: Defines how Route 53 responds to DNS queries, enabling complex traffic management strategies (source: sol-design.md).


- **Alias Records**: Can be used to route traffic to regional endpoints, such as an [[aws-api-gateway]] endpoint (source: sol-design.md).


## Related pages


- [[ai-solution-designs]]
- [[aws-api-gateway]]