# 1. Start from an official Python base image
# Using "-slim" keeps the image size smaller
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# 4. Install the Python dependencies
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Expose the port the app runs on
EXPOSE 80

# 7. Define the command to run your app
# Use 0.0.0.0 to make the server accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]