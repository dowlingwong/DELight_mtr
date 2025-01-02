# Dependency tree of Helix plugins

```mermaid
graph TD;
  FitResults-->NoisePSDs;
  FitResults-->Events;
  NoisePSDs-->NoiseEvents;
  Events-->UVTriggers;
  Events-->QPTriggers;
  Events-->ToyDataRawRecords;
  NoiseEvents-->UVTriggers;
  NoiseEvents-->QPTriggers;
  NoiseEvents-->ToyDataRawRecords;
  UVTriggers-->ToyDataRawRecords;
  QPTriggers-->ToyDataRawRecords;
  Triggers-->ToyDataRawRecords;
```

### Links to plugins' documentation pages
- [FitResults](plugins/FitResults.md)
- [NoisePSDs](plugins/NoisePSDs.md)
- [Events](plugins/Events.md)
- [NoiseEvents](plugins/NoiseEvents.md)
- [UVTriggers](plugins/UVTriggers.md)
- [QPTriggers](plugins/QPTriggers.md)
- [Triggers](plugins/Triggers.md)
- [ToyDataRawRecords](plugins/ToyDataRawRecords.md)
