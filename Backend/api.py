from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from datetime import datetime
import re
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Pydantic model for date input with validation
class DateRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Date format must be YYYY-MM-DD")

    @validator("date")
    def check_year_range(cls, value):
        dt = datetime.strptime(value, "%Y-%m-%d")
        if dt.year < 2009 or dt.year > 2024:
            raise ValueError("Year must be between 2009 and 2024")
        return value

# Function to determine the quarter
def get_quarter(date: str) -> str:
    dt = datetime.strptime(date, "%Y-%m-%d")
    year = dt.year
    quarter = (dt.month - 1) // 3 + 1
    return f"{year}q{quarter}"

@app.get("/")
def read_root():
    return {"The API is running. Use /get_quarter to get the year and quarter for a given date."}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon available"}


# API Endpoint to get year and quarter
@app.post("/get_quarter")
def get_year_quarter(date_request: DateRequest):
    quarter_str = get_quarter(date_request.date)
    return {"year_quarter": quarter_str}
