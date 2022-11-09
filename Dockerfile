# Base Python Image for the container.
FROM python:3.10
# Working directory for the container.
WORKDIR /code
# Copy the requirements
COPY requirements.txt /code/requirements.txt
# Install all the requirements.
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# Copy the application files.
COPY Main/api /code
COPY Main/methods /code
COPY Main/saved_models /code
COPY Main/temp /code
# Expose the port (Container)
EXPOSE 8000
# Execute the command.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]