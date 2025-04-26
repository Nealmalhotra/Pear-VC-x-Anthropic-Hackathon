FROM python:3.11-slim

WORKDIR /app

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock first for better caching
COPY Pipfile Pipfile.lock* ./
# Install dependencies using pipenv
RUN pipenv install --deploy --system

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "src.main"]
