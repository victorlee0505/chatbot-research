from pydantic import BaseModel, Field

class StudentDetails(BaseModel):
    student_id: str = Field(
        default="",
        description="The unique ID of the student.",
    )
    first_name: str = Field(
        default="",
        description="The first name of the student.",
    )
    last_name: str = Field(
        default="",
        description="The last name or surname of the student.",
    )
    email: str = Field(
        default="",
        description="An email address associated with the student.",
    )
    phone_number: str = Field(
        default="",
        description="The phone number of the student.",
    )
    address: str = Field(
        default="",
        description="The address where the student lives.",
    )
    graduation_year: int = Field(
        default=0,
        description="The year the student is expected to graduate.",
    )
    major: str = Field(
        default="",
        description="The major or field of study of the student.",
    )
    university: str = Field(
        default="",
        description="The name of the university where the student is enrolled.",
    )
    gpa: float = Field(
        default=0.0,
        description="The GPA (Grade Point Average) of the student.",
    )
    skills: str = Field(
        default="",
        description="Skills and abilities possessed by the student.",
    )
    experience: str = Field(
        default="",
        description="Relevant work experience of the student.",
    )
    education: str = Field(
        default="",
        description="Educational qualifications of the student.",
    )
    projects: str = Field(
        default="",
        description="Projects or assignments completed by the student.",
    )
    summary: str = Field(
        default="",
        description="A brief summary or description of the student.",
    )
