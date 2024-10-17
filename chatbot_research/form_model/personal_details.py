from pydantic import BaseModel, Field


class PersonalDetails(BaseModel):
    first_name: str = Field(
        ...,
        description="This is the first name of the user.",
    )
    last_name: str = Field(
        ...,
        description="This is the last name or surname of the user.",
    )
    full_name: str = Field(
        ...,
        description="Is the full name of the user ",
    )
    city: str = Field(
        ...,
        description="The name of the city where someone lives",
    )
    email: str = Field(
        ...,
        description="an email address that the person associates as theirs",
    )
    language: str = Field(
        ...,
        description="The language that the person speaks",
    )
