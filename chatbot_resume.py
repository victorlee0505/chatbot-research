from chatbot_research.form_model.student_details import StudentDetails
from chatbot_research.huggingface.chatbots.hf_resume_processor import (
    HuggingFaceChatbotResumeProcessor,
)
from chatbot_research.huggingface.config.hf_llm_config import HERMES_2_PRO_LLAMA_3_8B_Q8


def main():

    pydantic_model = StudentDetails

    bot = HuggingFaceChatbotResumeProcessor(
        llm_config=HERMES_2_PRO_LLAMA_3_8B_Q8,
        pydantic_model=pydantic_model,
        disable_mem=True,
    )

    students_detail = []

    folder_path = "./resume_input"
    staging_path = "./staging"

    bot.copy_pdfs_to_staging(src_folder=folder_path, staging_folder=staging_path)
    storages_path = bot.ingest_resumes(staging_path=staging_path)
    bot.create_model()
    for storage_path in storages_path:
        vectorstore = bot.create_vectorstore(storage_path)
        student_detail = bot.process_resume(vectorstore=vectorstore)
        students_detail.append(student_detail)

    print(students_detail)


if __name__ == "__main__":
    main()
