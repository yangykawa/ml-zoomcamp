# Use an official Python runtime as a parent image
FROM python:3.11

RUN pip install pipenv

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy --ignore-pipfile
COPY ["predict.py", "xgb_final_model.bin", "./"]

COPY ["predict-test.py", "./"]
EXPOSE 9696 
ENTRYPOINT ["pipenv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]