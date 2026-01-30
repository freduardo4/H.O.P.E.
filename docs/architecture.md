# System Architecture

```mermaid
graph TD
    subgraph "Desktop Client (WPF/.NET)"
        UI[User Interface]
        HA[Hardware Adapter]
        OfflineSync[Offline Sync Engine]
    end

    subgraph "Backend (NestJS)"
        API[API Gateway]
        Auth[Auth Service]
        Diag[Diagnostics Service]
        Wiki[Wiki-Fix Service]
    end

    subgraph "AI/ML Pipeline (Python)"
        Train[Training Service]
        Inference[Inference Engine]
    end

    subgraph "Data Store"
        PG[(PostgreSQL)]
        S3[Object Storage]
        Neo[(Neo4j Graph)]
    end

    UI --> HA
    UI --> OfflineSync
    OfflineSync <--> API
    
    API --> Auth
    API --> Diag
    API --> Wiki
    
    Diag --> PG
    Wiki --> Neo
    
    Train <--> PG
    Train <--> S3
    
    Inference --> API
```
