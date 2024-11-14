from pydantic import BaseModel, Field


class StudentDetails(BaseModel):
    student_id: str = Field(
        default="",
        description="The unique identifier of the student."
    )
    full_name: str = Field(
        default="",
        description="Full name of the student in 'Last Name, First Name' format."
    )
    email: str = Field(
        default="",
        description="The primary email address of the student."
    )
    address: str = Field(
        default="",
        description="The address where the student lives.",
    )
    university: str = Field(
        default="",
        description="The name of the university the student is attending."
    )
    program: str = Field(
        default="",
        description="The program or major the student is enrolled in (e.g., Computer Science, Information Technology)."
    )
    gpa: str = Field(
        default=0.0,
        description="The student's current GPA, used to assess academic standing."
    )
    technical_skills: str = Field(
        default=[],
        description="A list of technical skills or programming languages the student is proficient in (e.g., Python, Java, SQL)."
    )
    work_experience: str = Field(
        default=[],
        description="A summary of any previous work experience or internships, if any."
    )
    projects: str = Field(
        default=[],
        description="A brief description of projects the student has worked on that showcase their skills and initiative."
    )
    achievements: str = Field(
        default=[],
        description="Any notable achievements or awards the student has received."
    )
    portfolio_link: str = Field(
        default="",
        description="A link to the student's portfolio or GitHub, if available."
    )