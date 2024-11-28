from langchain_community.vectorstores.faiss import FAISS
from pydantic import BaseModel

from chatbot_research.form_model.student_details import StudentDetails
from chatbot_research.huggingface.chatbots.hf_resume_processor_faiss import (
    HuggingFaceChatbotResumeProcessorFaiss,
)
from chatbot_research.huggingface.config.hf_llm_config import HERMES_2_PRO_LLAMA_3_8B_Q8


def main():

    pydantic_model = StudentDetails

    bot = HuggingFaceChatbotResumeProcessorFaiss(
        llm_config=HERMES_2_PRO_LLAMA_3_8B_Q8,
        pydantic_model=pydantic_model,
        disable_mem=True,
    )

    students_details = []

    folder_path = "./resume_input"
    staging_path = "./staging"

    bot.copy_pdfs_to_staging(src_folder=folder_path, staging_folder=staging_path)
    storages_path = bot.ingest_resumes(staging_path=staging_path)
    bot.create_model()
    for storage_path in storages_path:
        vectorstore: FAISS | None = bot.create_vectorstore(storage_path)
        student_detail: BaseModel = bot.process_resume(vectorstore=vectorstore)
        students_details.append(student_detail)

    for student_detail in students_details:
        print(obj=student_detail.model_dump_json())


if __name__ == "__main__":
    main()
