# Response Schemas

## Overview

This directory contains JSON schemas for PocketGuide response structures, organized by version.

### Envelope vs Payload

- **Envelope** (v0): The top-level response structure required for all model outputs. Defines metadata fields: summary, assumptions, uncertainty_notes, next_steps, verification_steps, payload_type, and payload.
- **Payload** (v1): The structured content within `envelope.payload`. Schemas are versioned separately because payload types evolve independently of the envelope.

### Versioning Strategy

Schemas are versioned to support evolution while maintaining backward compatibility:

- **v0/** (current): Canonical response envelope schema
- **v1/** (current): Payload schemas for 4 content types (itinerary, checklist, decision_tree, procedure)
- **v2+** (future): Enhanced envelope or payload schemas as needs grow

**Why separate versions?**
- The envelope is relatively stable and unlikely to change structure significantly.
- Payloads are domain-specific and may evolve independently (e.g., adding more payload types, changing itinerary structure).
- Separating them allows payload evolution without forcing envelope changes.

### Directory Structure

```
schemas/
├── v0/
│   ├── response_envelope.schema.json  # Envelope schema (rarely changes)
│   └── README.md
└── v1/
    ├── itinerary.payload.schema.json
    ├── checklist.payload.schema.json
    ├── decision_tree.payload.schema.json
    ├── procedure.payload.schema.json
    └── __init__.py
```

### Usage

Load schemas in Python using `pathlib` and `jsonschema`:

```python
from pathlib import Path
import json
import jsonschema

# Load envelope schema (v0)
envelope_schema = json.loads(
    (Path(__file__).parent / "v0" / "response_envelope.schema.json").read_text()
)

# Load payload schema (v1 - example: itinerary)
itinerary_schema = json.loads(
    (Path(__file__).parent / "v1" / "itinerary.payload.schema.json").read_text()
)

# Validate
full_response = {...}  # With envelope
jsonschema.validate(full_response, envelope_schema)

payload = full_response["payload"]
jsonschema.validate(payload, itinerary_schema)
```

### Adding New Payload Types

To add a new payload type (e.g., `guide` or `alert`):
1. Create `v1/<new_type>.payload.schema.json` with appropriate structure
2. Add the type to `response_envelope.schema.json` payload_type enum (in v0)
3. Add fixtures and tests

