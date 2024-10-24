# from typing import Callable, Optional
# import os
# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.indexes import SearchIndexClient 
# from azure.search.documents import SearchClient
# from langchain.utils import get_from_env
# from azure.search.documents.indexes.models import (  
#     SearchIndex,  
#     SearchField,  
#     SearchFieldDataType,  
#     SimpleField,  
#     SearchableField,
#     ComplexField,
#     SearchIndex,  
#     SemanticConfiguration,  
#     PrioritizedFields,  
#     SemanticField,  
#     SemanticSettings,  
# )

# # Allow overriding field names for Azure Search
# FIELDS_ID = get_from_env(
#     key="AZURESEARCH_FIELDS_ID", env_key="AZURESEARCH_FIELDS_ID", default="id"
# )
# FIELDS_CONTENT = get_from_env(
#     key="AZURESEARCH_FIELDS_CONTENT",
#     env_key="AZURESEARCH_FIELDS_CONTENT",
#     default="content",
# )
# FIELDS_CONTENT_VECTOR = get_from_env(
#     key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
#     env_key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
#     default="content_vector",
# )
# FIELDS_METADATA = get_from_env(
#     key="AZURESEARCH_FIELDS_TAG", env_key="AZURESEARCH_FIELDS_TAG", default="metadata"
# )


# def get_search_client(
#     endpoint: str,
#     key: str,
#     index_name: str,
#     embedding_function: Callable,
#     semantic_configuration_name: Optional[str] = None,
#     semantic_config: SemanticConfiguration = None,
#     fields = [],
# ) -> SearchClient:
#     from azure.core.credentials import AzureKeyCredential
#     from azure.core.exceptions import ResourceNotFoundError
#     from azure.identity import DefaultAzureCredential
#     from azure.search.documents import SearchClient
#     from azure.search.documents.indexes import SearchIndexClient
#     from azure.search.documents.indexes.models import (
#         PrioritizedFields, SearchableField, SearchField, SearchFieldDataType,
#         SearchIndex, SemanticConfiguration, SemanticField, SemanticSettings,
#         SimpleField, VectorSearch, VectorSearchAlgorithmConfiguration)

#     if key is None:
#         credential = DefaultAzureCredential()
#     else:
#         credential = AzureKeyCredential(key)
#     index_client: SearchIndexClient = SearchIndexClient(
#         endpoint=endpoint, credential=credential
#     )
#     try:
#         index_client.get_index(name=index_name)
#     except ResourceNotFoundError:
#         # Fields configuration
#         fields = [
#             SimpleField(
#                 name=FIELDS_ID,
#                 type=SearchFieldDataType.String,
#                 key=True,
#                 filterable=True,
#             ),
#             SearchableField(
#                 name=FIELDS_CONTENT,
#                 type=SearchFieldDataType.String,
#                 searchable=True,
#                 retrievable=True,
#             ),
#             SearchField(
#                 name=FIELDS_CONTENT_VECTOR,
#                 type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
#                 searchable=True,
#                 dimensions=len(embedding_function("Text")),
#                 vector_search_configuration="default",
#             ),
#             SearchableField(
#                 name=FIELDS_METADATA,
#                 type=SearchFieldDataType.String,
#                 searchable=True,
#                 retrievable=True,
#             ),
#         ]
#         # Vector search configuration
#         vector_search = VectorSearch(
#             algorithm_configurations=[
#                 VectorSearchAlgorithmConfiguration(
#                     name="default",
#                     kind="hnsw",
#                     hnsw_parameters={
#                         "m": 4,
#                         "efConstruction": 400,
#                         "efSearch": 500,
#                         "metric": "cosine",
#                     },
#                 )
#             ]
#         )
#         # Create the semantic settings with the configuration
#         semantic_settings = (
#             None
#             if semantic_configuration_name is None
#             else SemanticSettings(
#                 configurations=[
#                     SemanticConfiguration(
#                         name=semantic_configuration_name,
#                         prioritized_fields=PrioritizedFields(
#                             prioritized_content_fields=[
#                                 SemanticField(field_name=FIELDS_CONTENT)
#                             ],
#                         ),
#                     )
#                 ]
#             )
#         )
#         # Create the search index with the semantic settings and vector search
#         index = SearchIndex(
#             name=index_name,
#             fields=fields,
#             vector_search=vector_search,
#             semantic_settings=semantic_settings,
#         )
#         index_client.create_index(index)
#     # Create the search client
#     return SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)