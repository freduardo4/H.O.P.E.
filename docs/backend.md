# HOPE Backend Architecture

## Overview
The HOPE Backend is a high-performance, modular API built with **NestJS** (Node.js). It serves as the central hub for the HOPE ecosystem, managing authentication, vehicle diagnostics, ECU calibrations, and the tuning marketplace.

**Key Technologies:**
- **Framework:** NestJS (Modular, Dependency Injection)
- **Language:** TypeScript
- **Database:** PostgreSQL 16 + TimescaleDB (Time-series data) + Neo4j (Knowledge Graph)
- **API:** REST + GraphQL (Apollo)
- **Documentation:** Swagger/OpenAPI (`/api`)

## Module Structure
The application is organized into domain-specific modules in `src/modules/`:

| Module | Description |
|--------|-------------|
| **Auth** | JWT-based authentication, Role-Based Access Control (RBAC), and OAuth2 SSO. |
| **Vehicles** | Fleet management, VIN decoding, and vehicle profiles. |
| **Diagnostics** | Ingestion of high-frequency OBD2/CAN data sessions. |
| **EcuCalibrations** | Version-controlled storage of binary ECU files (BIN/HEX). |
| **Marketplace** | Secure B2B/B2C exchange for tuning files with encryption. |
| **WikiFix** | Community-driven repair database powered by NLP. |
| **SafetyLogs** | Audit trail for critical operations (SafeFlash events). |

## API Governance & Quality

### 1. DTO Validation
We enforce strict data validation using `class-validator` and `class-transformer`. All input DTOs are validated globally.
- **Global Pipe:** `ValidationPipe` is enabled in `main.ts` with `whitelist: true` to strip invalid properties.
- **Example:**
  ```typescript
  export class CreateUserDto {
      @IsEmail()
      email: string;

      @MinLength(8)
      password: string;
  }
  ```

### 2. Error Handling
A standardized error handling strategy is implemented via the global `AllExceptionsFilter`.
- **Function:** Catches all exceptions, logs 500 errors, and returns a consistent JSON error response.
- **Response Format:**
  ```json
  {
    "statusCode": 400,
    "timestamp": "2024-01-29T12:00:00.000Z",
    "path": "/api/auth/login",
    "message": "Invalid credentials"
  }
  ```

### 3. API Documentation
Swagger (OpenAPI) is integrated for all REST endpoints.
- **Access:** Available at `/api` when running the server.
- **Features:** Interactive testing, schema exploration, and Bearer token auth support.

## Security
- **Authentication:** `JwtAuthGuard` protects private routes.
- **Authorization:** `RolesGuard` enforces `UserRole` (e.g., `ADMIN`, `TUNER`, `TECHNICIAN`).
- **Encryption:** Sensitive files (ECU binaries) are encrypted at rest using AES-256-GCM.

## Testing
- **Unit Tests:** Jest `.spec.ts` files co-located with services/controllers.
- **Mocking:** Heavy use of `jest.mock()` and `@nestjs/testing` for isolation.
- **Coverage:** Critical paths (Auth, Tuning, Marketplace) have high test coverage.

## Performance & Load Testing
We use **k6** to validate system stability and performance under load.

### Setup
Load tests are located in `infrastructure/load-tests/` and run via Docker Compose.

### Running Tests
1. **Smoke Test** (Sanity check):
   ```bash
   docker-compose -f infrastructure/docker/docker-compose.load.yml run k6 run /scripts/smoke-test.js
   ```
2. **Load Test** (Simulated traffic):
   ```bash
   docker-compose -f infrastructure/docker/docker-compose.load.yml run k6 run /scripts/load-test.js
   ```
3. **Stress Test** (Breaking point):
   ```bash
   docker-compose -f infrastructure/docker/docker-compose.load.yml run k6 run /scripts/stress-test.js
   ```

### Thresholds
- **Response Time:** 95% of requests should be under 500ms (Smoke) / 2000ms (Load).
- **Error Rate:** < 1% failure rate for HTTP requests.
