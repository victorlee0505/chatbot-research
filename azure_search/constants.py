# from typing import Callable, Optional
# from azure.search.documents.indexes.models import (
#     PrioritizedFields,
#     SearchableField,
#     SearchField,
#     SearchFieldDataType,
#     SearchIndex,
#     SemanticConfiguration,
#     SemanticField,
#     SemanticSettings,
#     SimpleField,
#     VectorSearch,
#     VectorSearchAlgorithmConfiguration,
# )

# # Content metadata properties used in Azure Cognitive Search
# FIELDS_ID = "id"
# FIELDS_CONTENT = "content"
# FIELDS_CONTENT_VECTOR = "content_vector"

# FIELDS_METADATA_AUTHOR = "metadata_author"
# FIELDS_METADATA_CHARACTER_COUNT = "metadata_character_count"
# FIELDS_METADATA_CONTENT_ENCODING = "metadata_content_encoding"
# FIELDS_METADATA_CONTENT_TYPE = "metadata_content_type"
# FIELDS_METADATA_CREATION_DATE = "metadata_creation_date"
# FIELDS_METADATA_DESCRIPTION = "metadata_description"
# FIELDS_METADATA_IDENTIFIER = "metadata_identifier"
# FIELDS_METADATA_KEYWORDS = "metadata_keywords"
# FIELDS_METADATA_LANGUAGE = "metadata_language"
# FIELDS_METADATA_LAST_MODIFIED = "metadata_last_modified"
# FIELDS_METADATA_MESSAGE_BCC = "metadata_message_bcc"
# FIELDS_METADATA_MESSAGE_BCC_EMAIL = "metadata_message_bcc_email"
# FIELDS_METADATA_MESSAGE_CC = "metadata_message_cc"
# FIELDS_METADATA_MESSAGE_CC_EMAIL = "metadata_message_cc_email"
# FIELDS_METADATA_MESSAGE_FROM = "metadata_message_from"
# FIELDS_METADATA_MESSAGE_FROM_EMAIL = "metadata_message_from_email"
# FIELDS_METADATA_MESSAGE_TO = "metadata_message_to"
# FIELDS_METADATA_MESSAGE_TO_EMAIL = "metadata_message_to_email"
# FIELDS_METADATA_PAGE_COUNT = "metadata_page_count"
# FIELDS_METADATA_PUBLISHER = "metadata_publisher"
# FIELDS_METADATA_SLIDE_COUNT = "metadata_slide_count"
# FIELDS_METADATA_SUBJECT = "metadata_subject"
# FIELDS_METADATA_TITLE = "metadata_title"
# FIELDS_METADATA_WORD_COUNT = "metadata_word_count"

# # https://learn.microsoft.com/en-us/azure/search/search-blob-metadata-properties
# FIELDS_MAPPING = {
#     "csv": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_CONTENT_ENCODING],
#     "doc": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CHARACTER_COUNT, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_PAGE_COUNT, FIELDS_METADATA_WORD_COUNT],
#     "docm": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CHARACTER_COUNT, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_PAGE_COUNT, FIELDS_METADATA_WORD_COUNT],
#     "docx": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CHARACTER_COUNT, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_PAGE_COUNT, FIELDS_METADATA_WORD_COUNT],
#     "eml": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_MESSAGE_FROM, FIELDS_METADATA_MESSAGE_TO, FIELDS_METADATA_MESSAGE_CC, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_SUBJECT],
#     "epub": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_TITLE, FIELDS_METADATA_DESCRIPTION, FIELDS_METADATA_LANGUAGE, FIELDS_METADATA_KEYWORDS, FIELDS_METADATA_IDENTIFIER, FIELDS_METADATA_PUBLISHER],
#     "gz": [FIELDS_METADATA_CONTENT_TYPE],
#     "html": [FIELDS_METADATA_CONTENT_ENCODING, FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_LANGUAGE, FIELDS_METADATA_DESCRIPTION, FIELDS_METADATA_KEYWORDS, FIELDS_METADATA_TITLE],
#     "json": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_CONTENT_ENCODING],
#     "kml": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_CONTENT_ENCODING, FIELDS_METADATA_LANGUAGE],
#     "msg": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_MESSAGE_FROM, FIELDS_METADATA_MESSAGE_FROM_EMAIL, FIELDS_METADATA_MESSAGE_TO, FIELDS_METADATA_MESSAGE_TO_EMAIL, FIELDS_METADATA_MESSAGE_CC, FIELDS_METADATA_MESSAGE_CC_EMAIL, FIELDS_METADATA_MESSAGE_BCC, FIELDS_METADATA_MESSAGE_BCC_EMAIL, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_SUBJECT],
#     "odp": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_TITLE],
#     "ods": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED],
#     "odt": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CHARACTER_COUNT, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_PAGE_COUNT, FIELDS_METADATA_WORD_COUNT],
#     "pdf": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_LANGUAGE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_TITLE, FIELDS_METADATA_CREATION_DATE],
#     "txt": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_CONTENT_ENCODING, FIELDS_METADATA_LANGUAGE],
#     "ppt": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_SLIDE_COUNT, FIELDS_METADATA_TITLE],
#     "pptm": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_SLIDE_COUNT, FIELDS_METADATA_TITLE],
#     "pptx": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_SLIDE_COUNT, FIELDS_METADATA_TITLE],
#     "rtf": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CHARACTER_COUNT, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED, FIELDS_METADATA_PAGE_COUNT, FIELDS_METADATA_WORD_COUNT],
#     "xls": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED],
#     "xlsm": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED],
#     "xlsx": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_AUTHOR, FIELDS_METADATA_CREATION_DATE, FIELDS_METADATA_LAST_MODIFIED],
#     "xml": [FIELDS_METADATA_CONTENT_TYPE, FIELDS_METADATA_CONTENT_ENCODING, FIELDS_METADATA_LANGUAGE],
#     "zip": [FIELDS_METADATA_CONTENT_TYPE]
# }

# class SearchFieldDataType:
#     String = "Edm.String"
#     Int32 = "Edm.Int32"
#     Int64 = "Edm.Int64"
#     Single = "Edm.Single"
#     Double = "Edm.Double"
#     Boolean = "Edm.Boolean"
#     DateTimeOffset = "Edm.DateTimeOffset"
#     GeographyPoint = "Edm.GeographyPoint"
#     ComplexType = "Edm.ComplexType"

#     def Collection(typ):
#         # type (str) -> str
#         return "Collection({})".format(typ)


# def get_fields_properties_for(ext: str, embedding_function: Callable):

#     if ext in FIELDS_MAPPING:
#         # Default Fields
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
#         ]

#         metadata_list = FIELDS_MAPPING.get(ext)
#         fields_list = []
#         for metadata in metadata_list:
#             field = SearchableField(
#                 name=metadata,
#                 type=SearchFieldDataType.String,
#                 searchable=True,
#                 filterable=False,
#                 facetable=False,
#                 sortable=False,
#                 retrievable=True,
#             )
#             fields_list.append(field)

#         return fields + fields_list
        
#     raise ValueError(f"Unsupported file extension '{ext}'")