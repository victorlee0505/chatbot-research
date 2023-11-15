from enum import Enum
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter


class LanguageExtension(str, Enum):
    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    JS = "js"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "py"
    RST = "rst"
    RUBY = "rjs"
    RUST = "rs"
    SCALA = "sc"
    SWIFT = "swift"
    # MARKDOWN = "md"
    LATEX = "tex"
    HTML = "html"


def get_language(language) -> Language:
    if language == LanguageExtension.CPP.value:
        return Language.CPP
    elif language == LanguageExtension.GO.value:
        return Language.GO
    elif language == LanguageExtension.JAVA.value:
        return Language.JAVA
    elif language == LanguageExtension.JS.value:
        return Language.JS
    elif language == LanguageExtension.PHP.value:
        return Language.PHP
    elif language == LanguageExtension.PROTO.value:
        return Language.PROTO
    elif language == LanguageExtension.PYTHON.value:
        return Language.PYTHON
    elif language == LanguageExtension.RST.value:
        return Language.RST
    elif language == LanguageExtension.RUBY.value:
        return Language.RUBY
    elif language == LanguageExtension.RUST.value:
        return Language.RUST
    elif language == LanguageExtension.SCALA.value:
        return Language.SCALA
    elif language == LanguageExtension.SWIFT.value:
        return Language.SWIFT
    # elif language == LanguageExtension.MARKDOWN.value:
    #     return Language.MARKDOWN
    elif language == LanguageExtension.LATEX.value:
        return Language.LATEX
    elif language == LanguageExtension.HTML.value:
        return Language.HTML
    else:
        raise ValueError(
            f"Language {language} is not supported! "
            f"Please choose from {list(Language)}"
        )


def codeTextSplitter(
    language: Language,
    chunk_size: int,
    chunk_overlap: int,
    documents: Iterable[Document],
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents(documents)
    return docs

