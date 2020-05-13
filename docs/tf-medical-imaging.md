# TF medical imaging design

Use case:
- Federated learning from radiology images
- Secure aggregation
- Single-use accountable prediction (encrypted inference)

Considerations
- Focus on modeling & crypto, not on infrastructure
- How to avoid picking up on spurious patterns, e.g. different
  data collection practices
- How should we allocate data to clients?
    - If we don't do this correctly, the results are much weaker
      :,)
