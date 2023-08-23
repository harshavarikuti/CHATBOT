# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory within the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('wordnet')"


# Copy the rest of the application code into the container at /app
COPY . /app/

# Expose port 5000 to the outside world
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP app.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]