from __future__ import annotations

import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureMachineLearningSkill,
    ConditionalSkill,
    CustomEntityLookupSkill,
    DocumentExtractionSkill,
    EntityLinkingSkill,
    EntityRecognitionSkill,
    EntityRecognitionSkillV3,
    ImageAnalysisSkill,
    KeyPhraseExtractionSkill,
    LanguageDetectionSkill,
    MergeSkill,
    OcrSkill,
    PIIDetectionSkill,
    SentimentSkill,
    SentimentSkillV3,
    ShaperSkill,
    SplitSkill,
    TextTranslationSkill,
    WebApiSkill,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerSkillset,
)

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_BASE_URL")  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # this may change in the future
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = "text-embedding-ada-002"

vector_store_address = os.getenv("AZURE_OPENAI_SEARCH_ENDPOINT")
vector_store_password = os.getenv("AZURE_OPENAI_SEARCH_ADMIN_KEY")


def get_blob_datasource_connection(
    endpoint: str,
    key: str,
    type: str,
    connection_string: str,
) -> SearchIndexerDataSourceConnection:
    if key is None:
        credential = DefaultAzureCredential()
    else:
        credential = AzureKeyCredential(key)

    indexer_client: SearchIndexerClient = SearchIndexerClient(
        endpoint=endpoint, credential=credential
    )
    container = SearchIndexerDataContainer(name="searchcontainer")
    data_source_connection = SearchIndexerDataSourceConnection(
        name=f"{index_name}-blob",
        type="azureblob",
        connection_string=connection_string,
        container=container,
    )
    data_source = indexer_client.create_or_update_data_source_connection(
        data_source_connection
    )

    if data_source is None:
        raise Exception("blob datasource is not connected")
    return data_source_connection


class AzureCognitiveEnrichment:
    def __init__(
        self,
        endpoint: str,
        key: str,
        index_name: str,
        data_source: SearchIndexerDataSourceConnection,
    ):
        """Initialize with necessary components."""
        # Initialize base class
        self.endpoint = endpoint
        self.key = key
        self.index_name = index_name
        self.skillset_name = f"{self.index_name}-skillset"
        self.indexer_name = f"{index_name}-indexer"
        self.data_source = data_source

        if self.endpoint is None or len(self.endpoint) == 0:
            raise ValueError(f"endpoint can not be null.")
        if self.index_name is None or len(self.index_name) == 0:
            raise ValueError(f"index_name can not be null.")
        if self.data_source is None or len(self.data_source.name) == 0:
            raise ValueError(f"data_source can not be null.")

    def get_search_indexer_client(
        self,
        skillset: SearchIndexerSkillset = None,
    ) -> SearchIndexerClient:
        if self.key is None:
            credential = DefaultAzureCredential()
        else:
            credential = AzureKeyCredential(self.key)

        indexer_client: SearchIndexerClient = SearchIndexerClient(
            endpoint=self.endpoint, credential=credential
        )

        # create_or_update_skillset for indexer
        if skillset:
            indexer_client.create_or_update_skillset(skillset)

        # Create an indexer
        indexer = SearchIndexer(
            name=self.indexer_name,
            description=f"Indexer for {self.index_name}",
            skillset_name=self.skillset_name,
            target_index_name=self.index_name,
            data_source_name=self.data_source.name,
            # field_mappings=[
            #     FieldMapping(source_field_name="metadata_storage_path", target_field_name="imageUrl"),
            #     FieldMapping(source_field_name="metadata_storage_name", target_field_name="title")
            # ],
            # output_field_mappings=[
            #     FieldMapping(source_field_name="/document/imageVector", target_field_name="imageVector")
            # ]
        )

        indexer_client = SearchIndexerClient(
            self.endpoint, AzureKeyCredential(self.key)
        )
        indexer_result = indexer_client.create_or_update_indexer(indexer)

        return indexer_client

    def get_search_skillset(
        self,
        azureml: bool = False,
        conditional: bool = False,
        custom_entity_lookup: bool = False,
        document_extraction: bool = False,
        entity_linking: bool = False,
        entity_recognition: bool = False,
        entity_recognition_v3: bool = False,
        image_analysis: bool = False,
        key_phrase_extraction: bool = False,
        lang_detection: bool = False,
        merge: bool = False,
        ocrskill: bool = False,
        pii_detection: bool = False,
        sentiment: bool = False,
        sentiment_v3: bool = False,
        shaper: bool = False,
        split: bool = False,
        text_translation: bool = False,
        webapi: bool = False,
    ) -> SearchIndexerSkillset:
        # Define the skillset
        ocr_search_indexer_skill = []

        if ocrskill:
            ocr_skill = OcrSkill(
                name="ocr-skill",
                description="Extract text from images using OCR",
                context="/document/normalized_images/*",
                default_language_code="en",
                should_detect_orientation=True,
            )
            ocr_search_indexer_skill.append(ocr_skill)

        # Define the skillset
        skillset = SearchIndexerSkillset(
            name=self.skillset_name, skills=[ocr_search_indexer_skill]
        )

        return skillset


# sample to run
index_name: str = "langchain-eng-demo"
blob_connection_string = "your_connection_string"
datasource = get_blob_datasource_connection(
    endpoint=vector_store_address,
    key=vector_store_password,
    connection_string=blob_connection_string,
)

enrichment = AzureCognitiveEnrichment(
    endpoint=vector_store_address,
    key=vector_store_password,
    index_name=index_name,
    data_source=datasource,
)

skillset = enrichment.get_search_skillset(ocrskill=True)

indexer_client = enrichment.get_search_indexer_client(skillset=skillset)

# this will run the indexer (scan new doc as well as run skill)
indexer_client.run_indexer(index_name)
