# Base Python Image for the container.
FROM python:3.9
# Working directory for the container.
WORKDIR /code
#
COPY ./app/requirements.txt /code/app/requirements.txt
#
RUN pip install --no-cache-dir --upgrade -r /code/app/requirements.txt
#
COPY ./app /code/app
#
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
