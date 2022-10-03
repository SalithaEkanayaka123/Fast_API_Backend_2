# Base Python Image for the container.
FROM python:3.10
# Working directory for the container.
WORKDIR /code
# Copy the requirements 
COPY ./app/requirements.txt /code/app/requirements.txt
# Install all the requirements.
RUN pip install --no-cache-dir --upgrade -r /code/app/requirements.txt
# Copy project files
COPY ./app /code/app
# Expose the port (Container)
EXPOSE 8000
# Execute the command.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
