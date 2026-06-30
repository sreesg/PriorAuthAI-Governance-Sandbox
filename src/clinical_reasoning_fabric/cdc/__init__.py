"""CDC Pipeline — Change Data Capture from Snowflake/Iceberg to Neo4j.

Projects current clinical state from source systems into the Causal Ontology Graph
via dbt transformation models and Dagster orchestration.
"""

from clinical_reasoning_fabric.cdc.cdc_pipeline_service import (
    BatchProcessingResult,
    CDCPipelineService,
    CheckpointStoreProtocol,
    FileCheckpointStore,
    GraphEntity,
    GraphEntityKind,
    InMemoryCheckpointStore,
    ProcessingResult,
    GraphServiceProtocol,
    DagsterClientProtocol,
    VALID_NODE_TYPES,
    VALID_RELATIONSHIP_TYPES,
)

__all__ = [
    "BatchProcessingResult",
    "CDCPipelineService",
    "CheckpointStoreProtocol",
    "FileCheckpointStore",
    "GraphEntity",
    "GraphEntityKind",
    "InMemoryCheckpointStore",
    "ProcessingResult",
    "GraphServiceProtocol",
    "DagsterClientProtocol",
    "VALID_NODE_TYPES",
    "VALID_RELATIONSHIP_TYPES",
]
